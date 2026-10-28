#' @title Generate a priority-screening ranking with a reliability guarantee
#'
#' @description
#' `.generate_prioritized_data` runs one iteration of the AI-assisted, target-based priority screening
#' workflow. It takes a corpus that has already been screened by an AI model and by humans, and
#' returns the records the human reviewers would still have to screen, sorted so that the records most
#' likely to be relevant come first. The ranking comes from a classifier trained on the text
#' embeddings of the records (see [embed_corpus()]).
#'
#' The goal is to save screening work without losing recall. Humans screen a development set of
#' records, which is used to build and validate the AI screener, and then
#' double-screen the records the AI included. The records the AI excluded are then screened in
#' priority order, and screening
#' can stop once every record in a small random *target set* has been found. Because the target set
#' is drawn with the sampling-with-replacement method of Hou and Tipton (2024), stopping at that
#' point gives a formal guarantee: the probability of reaching at least `c_target` recall is at
#' least `R_c`.
#'
#' @details
#' The following steps are the steps defined in Algorithm 1 which is implemented in this function. 
#' The steps are described in detail below:
#' 
#' 1. **Split off the finally included studies.** Records with `included_final == 1` are set aside.
#'    They are the ground truth, and some of them become seed studies.
#' 
#' 2. **Draw an irrelevant test set.** In the real workflow, humans screen a development set of
#'    records (the seed records plus a random sample, set V in the algorithm) to build and validate
#'    the AI screener. Those records are already human-screened, so they cost work but are not
#'    screened again. Here, `n_irrelevant_test_records` records from that set that both the AI and
#'    the humans excluded (`human_code == 0` and `decision_binary == 0`) are sampled and removed
#'    from the candidate pool. If `data` has no `human_code` column, records with
#'    `included_final == 0` and `decision_binary == 0` are used instead. They later serve as the irrelevant training examples (set E).
#' 
#' 3. **Optionally simulate AI misses.** A share `ai_miss_pct` of the finally included studies has its
#'    `decision_binary` set to 0, as if the AI had excluded them. These are called *AI-missed* records.
#'    Use this to test how robust the workflow is when the AI screener misses relevant records.
#' 
#' 4. **Pick seed studies.** A share `seed_pct` of the finally included studies the AI did catch
#'    become the seed studies. There can be few of them, or none (`seed_pct = 0`). Seeds are known
#'    to be relevant before screening starts.
#'    [Hou & Tipton (2024)](https://doi.org/10.1002/jrsm.1690) point out that such records are not a 
#'    random sample of the relevant records, so
#'    they must not be used as target records. They are only used to help train the classifier. The
#'    remaining finally included studies are put back into the candidate pool as ordinary records, so
#'    they can only be found through the screening process.
#' 
#' 5. **Define A- and AH+.** A- (AI-excluded records) are all records with `decision_binary == 0`,
#'    plus the AI-missed records. AH+ are the records that the AI and the humans both included, i.e.
#'    `included_var == 1`.
#' 
#' 6. **Draw the target set T.** [sample_target()] draws records from AH+ at random, with
#'    replacement, until `k` relevant records have been found. 
#' 
#' 7. **Split the seeds.** The seed studies are split into a training part (S_t, a share
#'    `seed_train_pct`) and a held-out validation part (S_v, the rest). By default
#'    (`seed_train_pct = 1`) all seed studies are used for training and none are held out. Use a
#'    lower value to keep some seed studies out of the training, as a test of the ranking.
#' 
#' 8. **Build the training data.** The included training set is `I = S_t + (AH+ minus T)`. The target
#'    records are left out on purpose, so the classifier never learns from them. The irrelevant
#'    training set E is sampled from the irrelevant test records, with the aim of having as many
#'    records as I.
#' 
#' 9. **Train the classifier.** The model is fitted on the embedding vectors of I (label 1) and E
#'    (label 0). With `alpha` between 0 and 1 this is a penalised logistic regression
#'    ([glmnet::cv.glmnet()]), where 0 is Ridge, 1 is LASSO and values in between are Elastic Net.
#'    With `alpha = 2` it is a probability random forest ([ranger::ranger()]).
#' 
#' 10. **Rank the priority-screening set.** `P* = A- + T + S_v` is scored by the model and sorted
#'     from the highest to the lowest priority score.
#'
#' **Using the result.** In practice, reviewers screen `P*` from the top and stop when all records
#' in T and S_v have been found (the original workflow also uses a 39 percent screening cap). The
#' columns `is_target`, `is_seed` and `is_ai_missed` mark these records, so the stopping point and
#' the workload saved can be read off directly from the output. This function does not stop
#' anywhere itself; it only ranks, and the stopping row is where the last target record sits.
#'
#' **Workload.** The attribute `screen_decisions_info` counts the human screening decisions. Every
#' record counts as two decisions because it is double-screened. The saved workload is
#' `(total - test - after AI - among target) / total`, where *total* is all records, *test* is the
#' irrelevant test records that humans screened while developing the AI screener, *after AI* is the
#' records the AI included (double-screened by humans), and *among target* is the non-target records
#' screened before the last target record is reached in the ranking.
#'
#' **Embeddings.** The embeddings are looked up by the record id (`id_col`) in a file made by
#' [embed_corpus()]. If that file does not exist yet for this data frame name and model, it is
#' created automatically the first time (this needs the title and abstract columns, see `title_col`
#' and `abstract_col`, and can take a while).
#'
#' **Randomness.** All random steps (test set, flipping, seeds, target sampling, training set E)
#' depend on the seed set by `seed`. Pass a number to get the same result every time. The exception
#' is the target sampling, which deliberately uses no fixed seed of its own, so that different
#' iterations give different target sets.
#'
#' @param data Data frame with one row per record. It must contain the record id column (`id_col`,
#'   `eppi_id` by default, unique), `included_final` (1 = finally included, 0 = not),
#'   `decision_binary` (AI screening decision, 1 = include) and the column named in `included_var`.
#'   The column `human_code` (human screening decision, 1 = include) is optional. Without it,
#'   `included_var` cannot be `"human_and_ai_in"` unless that column already exists.
#'
#'   Pass the data frame by its plain variable name, and use the same name as in [embed_corpus()]
#'   (e.g. `combined_data` in both). The name is used to find the saved embeddings.
#' @param model Character string with the embedding model that was used in [embed_corpus()], e.g.
#'   `"all-MiniLM-L6-v2"` or `"tfidf"`.
#' @param n_irrelevant_test_records Integer. Number of records that both the AI and the humans
#'   excluded, to be drawn as the irrelevant test set used for training. Default is 200.
#' @param included_var Character string naming the column that marks records included by both the
#'   AI and the human reviewers (AH+). Default is `"human_and_ai_in"`. The target set is sampled
#'   from the records where this column is 1.
#' @param c_target Numeric in (0, 1). The recall you want to reach, i.e. the share of all relevant
#'   records you want to find. `c_target = 0.95` means "find at least 95 out of every 100 relevant
#'   records". Default is 0.95. This is the goal for one screening project.
#' @param R_c Numeric in (0, 1). The reliability you want, i.e. how sure you want to be that
#'   `c_target` is really reached. Default is 0.95. The target set is drawn at random, so with bad
#'   luck a screening project can end below the goal. `R_c` is the chance that this does not happen.
#'   `R_c = 0.95` means a 95 percent chance of finding at least `c_target` of the relevant records,
#'   and a 5 percent (1 in 20) risk of finding fewer.
#'
#'   The chance of reaching `c_target` is at least `1 - c_target^k`. Since `k` is rounded up, this is
#'   usually a little higher than `R_c`.
#'
#'   The guarantee holds for relevant records like those the target set is drawn from (AH+). It
#'   covers records the AI missed only if the classifier ranks them like the target records. Use
#'   `ai_miss_pct` to test this.
#' @param alpha Numeric. Chooses the classifier. `0` is Ridge logistic regression (default), `1` is
#'   LASSO, a value between 0 and 1 is Elastic Net, and `2` uses a random forest instead.
#' @param ai_miss_pct Numeric between 0 and 1. Share of the finally included studies to flip to
#'   AI-missed, to simulate an AI screener that misses relevant records. Default is 0 (no flipping),
#'   and 1 means the AI missed all of them.
#' @param seed_pct Numeric between 0 and 1. Share of the AI-caught finally included studies used as
#'   seed studies. Default is 1 (all of them). Use a lower value if you have few good seed studies,
#'   and 0 for no seed studies at all (then only the records the AI and the humans agree on, minus
#'   the target, are used to train the classifier). The studies that are not used as seeds go back
#'   into the candidate pool as ordinary records.
#' @param seed_train_pct Numeric between 0 and 1. Share of the seed studies used for training the
#'   classifier (S_t). The rest are held out (S_v) and are added to the ranked set as extra
#'   validation records. Default is 1 (all seed studies are used for training, none are held out).
#'   Lower it, for example to 0.5, to keep some seed studies out of the training as a test.
#' @param embed_dir Character string with the folder where the embeddings are saved and read.
#'   Default is the current working directory. If no embeddings exist yet for this data frame and
#'   model, they are created here with [embed_corpus()] (`data` then needs the title and abstract
#'   columns). This can take a while, so if you run this function many times, run
#'   [embed_corpus()] once first.
#' @param seed Integer or NULL. Random seed for reproducibility. Default is NULL (no seed set).
#' @param id_col Character string with the name of the unique record id column. Default is
#'   "eppi_id". The same column must have been used in [embed_corpus()].
#' @param title_col Character string with the name of the title column. Default is "title". Only
#'   used if the embeddings have to be created.
#' @param abstract_col Character string with the name of the abstract column. Default is "abstract".
#'   Only used if the embeddings have to be created.
#' @param data_name Character string or NULL. Name used to find the saved embeddings. Default is
#'   NULL, which uses the name of the data frame you pass in. You normally do not need to set this.
#'
#' @return A data frame with the priority-screening set `P*` (AI-excluded records, the target
#'   records and the held-out seed studies), sorted by `priority_score` from highest to lowest. It
#'   has the original columns of `data` plus:
#'   \item{priority_score}{The classifier's predicted probability that the record is relevant.}
#'   \item{row_number}{The position in the ranking (1 = screened first).}
#'   \item{is_target}{1 if the record is in the target set T, otherwise 0.}
#'   \item{is_seed}{1 if the record is a held-out seed study (S_v), otherwise 0.}
#'   \item{is_ai_missed}{1 if the record was flipped to AI-missed, otherwise 0.}
#'
#'   The result also carries these attributes:
#'   \item{total_records}{Number of records in `data`.}
#'   \item{total_true_relevant}{Number of finally included studies.}
#'   \item{known_relevant_n}{Number of relevant records already known through the training set I.}
#'   \item{info_dat}{A one-row tibble with the settings used (data name, model, classifier type,
#'     `c_target`, `R_c`, `alpha`, seed shares, `included_var`, `ai_miss_pct`) and the run time in
#'     seconds.}
#'   \item{screen_decisions_info}{A one-row tibble with the number of screening decisions in each
#'     part of the workflow and the proportion of workload saved (`workload_saved`).}
#'
#'   The function stops with an error if no target records are found.
#'
#' @references
#' Hou, Z., & Tipton, E. (2024). Enhancing recall in automated record screening:
#'   A resampling algorithm. \emph{Research Synthesis Methods, 15}(3), 372-383.
#'   \doi{10.1002/jrsm.1690}
#'
#' @examples
#' \dontrun{
#' # 1. Embed the corpus once (needs an id, a title and an abstract column)
#' embed_corpus(combined_data, model = "all-MiniLM-L6-v2", dir = "embeddings")
#'
#' # 2. Rank the records that still need to be screened
#' ranked <- .generate_prioritized_data(
#'   data = combined_data,
#'   model = "all-MiniLM-L6-v2",
#'   c_target = 0.95,
#'   R_c = 0.95,
#'   embed_dir = "embeddings",
#'   seed = 123
#' )
#'
#' # 3. How much workload was saved?
#' attr(ranked, "screen_decisions_info")$workload_saved
#'
#' # Stress test: pretend the AI missed 10 percent of the relevant studies
#' ranked_miss <- .generate_prioritized_data(
#'   data = combined_data,
#'   model = "all-MiniLM-L6-v2",
#'   ai_miss_pct = 0.1,
#'   embed_dir = "embeddings"
#' )
#' }
#' @keywords internal

.generate_prioritized_data <-
    function(
      data,
      model = "all-MiniLM-L6-v2",
      n_irrelevant_test_records = 200,
      included_var = "human_and_ai_in",
      c_target      = 0.95, 
      R_c           = 0.95,
      alpha         = 0, 
      ai_miss_pct   = 0,
      seed_pct      = 1,
      seed_train_pct = 1,
      embed_dir     = NULL,
      seed          = NULL,
      id_col        = "eppi_id",
      title_col     = "title",
      abstract_col  = "abstract",
      data_name     = NULL
  ) {


  # The embedding file is found through the name of the data frame.
  if (is.null(data_name)) data_name <- paste(deparse(substitute(data)), collapse = "")

  ######################################################
  # Initial checks
  ######################################################
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.")
  }

  if (!is.character(id_col) || length(id_col) != 1) {
    stop("`id_col` must be a single column name.")
  }

  required_cols <- c(id_col, "included_final", "decision_binary")
  if (!all(required_cols %in% colnames(data))) {
    stop("`data` must contain columns: ", paste(required_cols, collapse = ", "))
  }

  # human_code is optional. Without it, "excluded by the humans" is taken from included_final == 0.
  has_human_code <- "human_code" %in% colnames(data)

  if (included_var == "human_and_ai_in" && !included_var %in% colnames(data) && has_human_code) {
    data[[included_var]] <- as.integer(data[["human_code"]] == 1 & data[["decision_binary"]] == 1)
  }

  if (!is.character(included_var) || length(included_var) != 1 || !included_var %in% colnames(data)) {
    stop("`included_var` must be the name of a column in `data`.",
         if (!has_human_code) " Without a `human_code` column, `human_and_ai_in` cannot be made, so use e.g. included_var = \"decision_binary\"." else "",
         call. = FALSE)
  }

  # The name becomes part of the file name, so it must be a plain variable name
  if (grepl("[^A-Za-z0-9._-]", data_name)) {
    stop("Pass the data frame by its plain variable name (e.g. `combined_data`), not as an expression like `",
         data_name, "`. The name is used to find the saved embeddings.", call. = FALSE)
  }

  if (!is.character(model) || length(model) != 1) {
    stop("`model` must be a single character string.")
  }

  if (!is.numeric(n_irrelevant_test_records) || length(n_irrelevant_test_records) != 1 || n_irrelevant_test_records < 1) {
    stop("`n_irrelevant_test_records` must be a positive number.")
  }

  if (!is.numeric(c_target) || c_target <= 0 || c_target >= 1) {
    stop("`c_target` must be a numeric value in (0, 1).")
  }

  if (!is.numeric(R_c) || R_c <= 0 || R_c >= 1) {
    stop("`R_c` must be a numeric value in (0, 1).")
  }

    # Ensure that alpha is either btween 0 and 1, or 2 for the classifier choice
  if (!is.numeric(alpha) || !(alpha >= 0 && alpha <= 1 || alpha == 2)) {
    stop("`alpha` must be a numeric value in [0, 1] or equal to 2.")
  }

  if (!is.numeric(ai_miss_pct) || ai_miss_pct < 0 || ai_miss_pct > 1) {
    stop("`ai_miss_pct` must be a numeric value in [0, 1].")
  }

  if (!is.numeric(seed_pct) || seed_pct < 0 || seed_pct > 1) {
    stop("`seed_pct` must be a numeric value in [0, 1].")
  }

  if (!is.numeric(seed_train_pct) || seed_train_pct < 0 || seed_train_pct > 1) {
    stop("`seed_train_pct` must be a numeric value in [0, 1].")
  }

  if (is.null(embed_dir)) {
    embed_dir <- getwd() # look in the current working directory if no directory is given
  } else if (!is.character(embed_dir)) {
    stop("`embed_dir` must be a character string.")
  }

  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1)) {
    stop("`seed` must be a single numeric value.")
  }

  # Create the embeddings if they do not exist yet
  model_file <- gsub("/", "--", model, fixed = TRUE)
  if (!file.exists(file.path(embed_dir, paste0(data_name, "_", model_file, ".rds")))) {
    message("No embeddings found for '", data_name, "' with model '", model, "'. Creating them now. This can take a while.")
    embed_corpus(data, model = model, dir = embed_dir, id_col = id_col, title_col = title_col,
                 abstract_col = abstract_col, data_name = data_name)
  }

  total_records <- nrow(data)
  
  ## For workload saving computation ##
  total_screen_decisions <- total_records * 2

  # Embeddings are a deterministic function of (data, model) so they
  # are looked up here
  embeddings <- .load_embeddings(data_name, model, dir = embed_dir)

  # Every record must have an embedding, otherwise the file is out of date or belongs to other data
  missing_ids <- setdiff(as.character(data[[id_col]]), rownames(embeddings))
  if (length(missing_ids) > 0) {
    stop(length(missing_ids), " of the ", nrow(data), " records have no embedding (e.g. ", id_col, " ",
         paste(utils::head(missing_ids, 3), collapse = ", "), ").\n",
         "The saved embeddings in '", embed_dir, "' for '", data_name, "' and model '", model,
         "' do not match `data`. Delete that .rds file and run again to create new embeddings.",
         call. = FALSE)
  }

  run_start_time <- Sys.time()

  # One seed governs every random draw below; nothing downstream re-seeds
  if (!is.null(seed)) set.seed(seed)

  # Split off the finally included studies (included_final == 1) from the rest of the candidate pool
  final_inc_studies <- data |> dplyr::filter(included_final == 1)
  data              <- data |> dplyr::filter(included_final == 0)

  # Ground truth
  total_true_relevant <- nrow(final_inc_studies)

  # Draw the irrelevant test set only from records both AI and human excluded. At this point `data`
  # holds only records with included_final == 0, so without human_code the AI decision is enough.
  if (has_human_code) {
    irrelevant_pool_idx <- which(data[["human_code"]] == 0 & data[["decision_binary"]] == 0)
  } else {
    irrelevant_pool_idx <- which(data[["decision_binary"]] == 0)
  }
  if (length(irrelevant_pool_idx) < n_irrelevant_test_records) {
    stop("Only ", length(irrelevant_pool_idx), " records were excluded by both the AI and the humans, ",
         "but n_irrelevant_test_records is ", n_irrelevant_test_records, ". Lower n_irrelevant_test_records.",
         call. = FALSE)
  }
  irrelevant_test_study_idx <- sample(irrelevant_pool_idx, size = n_irrelevant_test_records)
  irrelevant_test_study <- data[irrelevant_test_study_idx, , drop = FALSE]

  ## Number of screener decisions needed to be made to create this data ##
  n_screen_decisons_test <- nrow(irrelevant_test_study) * 2    
      
  data <- data[-irrelevant_test_study_idx, , drop = FALSE]
      
  # Artificially flip decision_binary to 0 for a share of ALL the finally included studies
  n_flip <- round(ai_miss_pct * nrow(final_inc_studies))

  if (n_flip > 0L) {
    flip_idx <- sample.int(nrow(final_inc_studies), size = n_flip)
    final_inc_studies[["decision_binary"]][flip_idx] <- 0L
    ai_missed <- final_inc_studies[flip_idx, , drop = FALSE]
    caught_final_inc <- final_inc_studies[-flip_idx, , drop = FALSE]
  } else {
    flip_idx <- integer(0)
    ai_missed <- final_inc_studies[FALSE, , drop = FALSE]
    caught_final_inc <- final_inc_studies
  }

  # Extract seed studies as a percentage of the finally included studies the AI caught.
  # There can be few seed studies, or none: with seed_pct = 0 (or when the share rounds to 0)
  # the classifier is trained without seed studies.
  n_seed <- as.integer(round(seed_pct * nrow(caught_final_inc)))

  if (n_seed == 0L) {
    known_seed_studies <- caught_final_inc[FALSE, , drop = FALSE]
    non_seed_final_inc <- caught_final_inc
  } else {
    # The training part of the seeds needs at least 1 study. If some seeds are held out
    # (seed_train_pct < 1), the held-out part needs at least 1 study too.
    n_seed_train <- round(seed_train_pct * n_seed)
    if (n_seed_train < 1 || (seed_train_pct < 1 && n_seed_train > n_seed - 1)) {
      stop("With ", n_seed, " seed studies and seed_train_pct = ", seed_train_pct, ", ", n_seed_train,
           " would be used for training and ", n_seed - n_seed_train, " held out. ",
           "Training needs at least 1, and if some are held out, so does the held-out part. ",
           "Change seed_pct or seed_train_pct (1 uses all seeds for training).", call. = FALSE)
    }

    seed_idx <- sample.int(nrow(caught_final_inc), size = n_seed)
    known_seed_studies <- caught_final_inc[seed_idx, , drop = FALSE]
    non_seed_final_inc <- caught_final_inc[-seed_idx, , drop = FALSE]
  }

  # Candidate pool P = data, with the non-seed finally included studies folded back in as ordinary records
  data <- 
    data |> 
    dplyr::bind_rows(non_seed_final_inc) 

  # Step 6: Let 𝐀+ denote records classified as potentially eligible by 𝒜, and let 𝐀− denote records not classified as potentially eligible by 𝒜.
  a_minus <- dplyr::bind_rows(data |> dplyr::filter(.data[["decision_binary"]] == 0), ai_missed)

  # Step 7: Define 𝐀𝐇+ as all non-seed records with included_var == 1 up to this point.
  ah_plus <- data |> dplyr::filter(.data[[included_var]] == 1)
  if (nrow(ah_plus) == 0) {
    stop("No records have ", included_var, " == 1 outside the seed studies, so no target set can be drawn.",
         call. = FALSE)
  }

  # Number of human screening decisions needed to be made (humans double-screen all of A+)
  included_ai_dat <- data |> dplyr::filter(.data[["decision_binary"]] == 1)
  n_screen_decision_after_ai_screen <- nrow(included_ai_dat) * 2

  # Steps 8-14: target set T, sampled with replacement from AH+ until k_min relevant records are
  # found (Hou & Tipton)
  target <- sample_target(
    data = data,
    relevant_col = included_var,
    c_target = c_target,
    R_c = R_c,
    id_col = id_col,
    seed = NULL # Set seed to null to allow for variance across simulations
  ) |>
    suppressWarnings()

  target_ids <- target$target_ids

  # Step 15: Randomly split the correctly-caught seed studies into a training set 𝐒t and a
  # held-out validation set 𝐒v.
  St_idx <- sample.int(nrow(known_seed_studies), size = round(seed_train_pct * nrow(known_seed_studies)))
  St <- known_seed_studies[St_idx, ]
  Sv <- known_seed_studies[-St_idx, ]

  # Step 16: Define the included training set as: 𝐈 = 𝐒t%∪(𝐀𝐇+\ 𝐓)
  I_set <- dplyr::bind_rows(St, ah_plus |> dplyr::filter(!.data[[id_col]] %in% target_ids))

  # Of the truly relevant studies, how many are already known via the training set I_set
  known_relevant_n <- sum(I_set[["included_final"]] == 1, na.rm = TRUE)

  # Step 17: Randomly sample an irrelevant training set 𝐄    
  E_set <- 
    irrelevant_test_study |>
    dplyr::slice_sample(
      n = nrow(I_set),
      replace = FALSE
    )

  # Embeddings are precomputed over the whole corpus, so every record an iteration can draw is
  # already present; rows are pulled by record id. Duplicated ids (E_set is sampled with
  # replacement) resolve to the same row, as before.
  emb_rows <- function(ids) {
    idx <- match(as.character(ids), rownames(embeddings))
    embeddings[idx, , drop = FALSE]
  }

  # Step 18: Train ℳ using the included training set 𝐈 and the irrelevant training set 𝐄.
  train_ids <- c(I_set[[id_col]], E_set[[id_col]])
  x_train   <- emb_rows(train_ids)
  y_train   <- c(rep(1L, nrow(I_set)), rep(0L, nrow(E_set)))

  if (alpha == 2) {
    fit <- ranger::ranger(x = x_train, y = factor(y_train, levels = c(0, 1)), probability = TRUE)
  } else {
    fit <- glmnet::cv.glmnet(x = x_train, y = y_train, family = "binomial", alpha = alpha)
  }

  # Step 19: Define the priority-screening set as: 𝐏∗ = 𝐀−∪ 𝐓 ∪ 𝐒v%
  ## We need to remove the E studies from a_minus here.    
      
  #a_minus <- a_minus |> dplyr::filter(!.data[["eppi_id"]] %in% E_set[["eppi_id"]])
      
  P_star <- dplyr::bind_rows(a_minus, target$target_set, Sv) |>
    dplyr::distinct(.data[[id_col]], .keep_all = TRUE)

  # Step 20: Use ℳ to rank all records in 𝐏∗.
  x_pstar <- emb_rows(P_star[[id_col]])
  
  if (alpha == 2) {
    P_star$priority_score <- stats::predict(fit, data = x_pstar)$predictions[, "1"]
  } else {
    P_star$priority_score <- as.numeric(stats::predict(fit, newx = x_pstar, s = "lambda.min", type = "response"))
  }
  
  P_star <- P_star[order(-P_star$priority_score), ]   
  P_star$row_number     <- seq_len(nrow(P_star))
        
  #P_star$is_target      <- P_star[["eppi_id"]] %in% target_ids
  #P_star$is_seed <- P_star[["eppi_id"]] %in% Sv[["eppi_id"]]
  #P_star$is_ai_missed   <- P_star[["eppi_id"]] %in% ai_missed[["eppi_id"]]

  # Get the run time in seconds
  run_time_sec <- as.numeric(difftime(Sys.time(), run_start_time, units = "secs"))
       
  attr(P_star, "total_records") <- total_records
  attr(P_star, "total_true_relevant") <- total_true_relevant
  attr(P_star, "known_relevant_n") <- known_relevant_n

  attr(P_star, "info_dat") <- tibble::tibble(
      data_name = data_name,
      model = model, 
      c_target = c_target,
      R_c = R_c,
      alpha = alpha,
      seed_pct = seed_pct,
      seed_train_pct = seed_train_pct,
      included_var = included_var,
      ai_miss_pct = ai_miss_pct,
      run_time_sec = run_time_sec
  ) |> 
    dplyr::mutate(
      train_model = dplyr::case_when(
        alpha == 2 ~ "Random Forest",
        alpha == 1 ~ "LASSO",
        alpha == 0 ~ "Ridge",
        alpha > 0 & alpha < 1 ~ "Elastic Net",
        .default = "Check model"
      )
    ) |> 
    dplyr::relocate(train_model, .after = model)
  
      
  res <- 
    P_star |> 
    dplyr::mutate(
      is_target = dplyr::if_else(.data[[id_col]] %in% target_ids, 1L, 0L),
      is_seed = dplyr::if_else(.data[[id_col]] %in% Sv[[id_col]], 1L, 0L),
      is_ai_missed = dplyr::if_else(.data[[id_col]] %in% ai_missed[[id_col]], 1L, 0L)
    )
  
  target_rows <- res$row_number[res$is_target == 1L]

  if (length(target_rows) == 0L) {
    stop("No target rows found.", call. = FALSE)
  }

  last_target_row <- max(target_rows, na.rm = TRUE)
  within_target <- seq_len(last_target_row)

  among_target_dat <- res[
    within_target[
      !is.na(res$is_target[within_target]) &
        res$is_target[within_target] != 1L
    ],
    ,
    drop = FALSE
  ]
  
  ## Number of human screening decisions to be made in the target sample  
  n_screen_decisions_in_target_dat <- nrow(among_target_dat)
  
  attr(res, "screen_decisions_info") <- 
    tibble::tibble(
      total_decisions = total_screen_decisions,
      test_decisions = n_screen_decisons_test,
      after_ai_decisions = n_screen_decision_after_ai_screen,
      among_target_decisions = n_screen_decisions_in_target_dat,
      workload_saved = (total_decisions - test_decisions - after_ai_decisions - among_target_decisions)/total_decisions
    )    
      
  res 
      
}