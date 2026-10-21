#' @title Simulate target-based priority screening for one design and keep the best runs
#'
#' @description
#' `targetscreen` runs the priority-screening workflow `iterations` times for ONE design (one set
#' of `alpha`, `R_c`, `c_target` and so on). Only the random seed changes between iterations. It
#' returns a performance summary over all iterations, and the prioritized data of the best runs.
#'
#' Saving the prioritized data of every iteration would be heavy. Instead, each iteration only
#' stores its seed and a few numbers. The best runs are found from these numbers, and then run
#' again with their seeds to get their prioritized data. This works because the same seed always
#' gives the same prioritized data.
#'
#' Iteration `i` uses the seed `seed + i`. Raising `iterations` adds new runs and leaves the
#' existing ones unchanged.
#'
#' @details
#' **What is a best run?** A run is only a candidate if it reached the goal, i.e. if
#' `recall_pstar >= c_target`. This leaves out lucky runs that saved a lot of work by missing
#' relevant studies. Among the candidates, the runs with the highest `workload_saved` are the best.
#' `keep_pct` sets how many of the candidates are kept: `0.05` keeps the best 5 percent.
#'
#' **Check.** After the rerun, `workload_saved` is compared with the number stored in the first
#' pass. If they differ, there is a warning. It would mean that something other than the seed
#' (for example different data or embeddings) changed the result.
#'
#' @inheritParams .generate_prioritized_data
#' @param iterations Integer. Number of times to run the design. Default is 1000.
#' @param seed Integer. Start value for the seeds. Iteration `i` uses `seed + i`. The seeds are
#'   what lets the best runs be recreated. If no `seed` is given, `123` is used and there is a
#'   warning.
#' @param keep_pct Numeric in (0, 1]. Share of the runs that reached the goal to keep, counted from
#'   the best. `0.01` keeps the best 1 percent and `0.05` the best 5 percent. At least one run is
#'   always kept. Default is 0.05.
#' @param python_dir Character string with the path to the Python executable, or NULL. Only used
#'   if the embeddings do not exist yet and `model` is a sentence-transformers model. If NULL, an
#'   installed Python with sentence-transformers is looked for, see [embed_corpus()].
#'
#' @return A list with:
#'   \item{summary}{A one-row data frame with the design, the number of iterations
#'     (`n_sim`), the mean recall (`recall_pstar_mean`), the share of runs that reached `c_target`
#'     (`reliability_pstar` and its standard error), whether that share is at least `R_c`
#'     (`meets_guarantee`), the mean workload saved (`wl_mean`, `wl_se`), the mean share of
#'     records needed to find the target (`need_see_mean`, `need_see_se`) and the number of kept
#'     runs (`n_kept`).}
#'   \item{iterations}{A data frame with one row per iteration: `iteration`, `seed`,
#'     `workload_saved`, `recall_pstar`, `pct_needed_to_find_target`, `reached_goal` and `kept`.}
#'   \item{best_seeds}{The seeds of the kept runs, best first.}
#'   \item{best}{A named list with the prioritized data of the kept runs (named by seed), best
#'     first.}
#'
#' @examples
#' \dontrun{
#' embed_corpus(combined_data, model = "tfidf", dir = "embeddings")
#'
#' res <- targetscreen(
#'   data = combined_data,
#'   model = "tfidf",
#'   c_target = 0.90,
#'   R_c = 0.95,
#'   alpha = 0,
#'   embed_dir = "embeddings",
#'   iterations = 500,
#'   seed = 123,
#'   keep_pct = 0.05
#' )
#'
#' res$summary      # performance over all iterations
#' res$best_seeds   # seeds of the best 5 percent
#' res$best[[1]]    # prioritized data of the very best run
#' }
#' @export
targetscreen <-
  function(
    data,
    model = "all-MiniLM-L6-v2",
    n_irrelevant_test_records = 200,
    included_var = "human_and_ai_in",
    c_target = 0.95,
    R_c = 0.95,
    alpha = 0,
    ai_miss_pct = 0,
    seed_pct = 1,
    seed_train_pct = 0.5,
    embed_dir = NULL,
    iterations = 1000,
    seed,
    keep_pct = 0.05,
    python_dir = NULL
  ) {

  # The embedding file is found through the name of the data frame, so pass it on by name
  data_name <- paste(deparse(substitute(data)), collapse = "")

  ######################################################
  # Initial checks
  ######################################################
  if (missing(seed)) {
    warning("No `seed` was given. Using 123 as the default seed.")
    seed <- 123
  }

  if (!is.numeric(seed) || length(seed) != 1) {
    stop("`seed` must be a single numeric value. The seeds are needed to recreate the best runs.")
  }

  if (!is.numeric(iterations) || length(iterations) != 1 || iterations < 1) {
    stop("`iterations` must be a positive number.")
  }

  if (!is.numeric(keep_pct) || length(keep_pct) != 1 || keep_pct <= 0 || keep_pct > 1) {
    stop("`keep_pct` must be a numeric value in (0, 1].")
  }

  # Everything that is the same in every iteration. Only the seed changes. Therefore, seed is not included here.
  design <- list(
    data = data, 
    data_name = data_name, 
    model = model,
    n_irrelevant_test_records = n_irrelevant_test_records,
    included_var = included_var, 
    c_target = c_target, 
    R_c = R_c, 
    alpha = alpha,
    ai_miss_pct = ai_miss_pct, 
    seed_pct = seed_pct, 
    seed_train_pct = seed_train_pct,
    embed_dir = embed_dir
  )

  # Embed the corpus once, before the iterations. If the embeddings already exist this is skipped.
  # It is done here, not inside the iterations, as their messages are muted and it can take a while.
  embed_corpus(data, model = model, python_dir = python_dir, dir = embed_dir, data_name = data_name)

  ######################################################
  # Pass 1: run every iteration, keep only the seed and the numbers
  ######################################################
  # Iteration j always gets seed + j, so raising `iterations` adds replicates without changing
  # the existing ones
  iteration_seeds <- seed + seq_len(iterations)

  iteration_results <- lapply(seq_len(iterations), function(i) {
    s <- .summarise_run(.run_iteration(iteration_seeds[[i]], design)) # Get the metrics for this iteration to compare with other iterations
    tibble::tibble(
      iteration = i, seed = iteration_seeds[[i]],
      workload_saved = s$workload_saved, recall_pstar = s$recall_pstar,
      pct_needed_to_find_target = s$pct_needed_to_find_target
    )
  })
  results <- do.call(rbind, iteration_results)

  ######################################################
  # Pick the best runs: goal reached first, then highest workload saved
  ######################################################
  results$reached_goal <- !is.na(results$recall_pstar) & results$recall_pstar >= c_target

  candidate_idx <- which(results$reached_goal & !is.na(results$workload_saved))
  n_keep <- if (length(candidate_idx) == 0L) 0L else max(1L, ceiling(keep_pct * length(candidate_idx)))

  # Highest workload_saved first; ties keep the order of the iterations
  best_idx <- candidate_idx[order(-results$workload_saved[candidate_idx])][seq_len(n_keep)]
  results$kept <- seq_len(nrow(results)) %in% best_idx

  if (n_keep == 0L) {
    warning("No iteration reached the goal (recall_pstar >= c_target), so no best runs were kept.",
            call. = FALSE)
  }

  ######################################################
  # Pass 2: run the best seeds again to get the prioritized data
  ######################################################
  best_seeds <- results$seed[best_idx]

  best <- lapply(best_seeds, .run_iteration, design = design)
  names(best) <- best_seeds

  # Same seed should give the same result as in pass 1
  for (k in seq_along(best_idx)) {
    stored <- results$workload_saved[best_idx[[k]]]
    rerun <- attr(best[[k]], "screen_decisions_info")$workload_saved
    if (!isTRUE(all.equal(stored, rerun))) {
      warning("Seed ", best_seeds[[k]], " gave workload_saved ", rerun, " on the rerun but ", stored,
              " in the first pass. Something other than the seed changed the result.", call. = FALSE)
    }
  }

  ######################################################
  # Summary over all iterations
  ######################################################
  reliability <- mean(results$reached_goal)

  summary <- tibble::tibble(
    data_name = data_name, model = model, included_var = included_var,
    c_target = c_target, R_c = R_c, alpha = alpha,
    ai_miss_pct = ai_miss_pct, seed_pct = seed_pct, seed_train_pct = seed_train_pct,
    n_sim = iterations,
    recall_pstar_mean = mean(results$recall_pstar, na.rm = TRUE),
    reliability_pstar = reliability,
    reliability_pstar_se = sqrt(reliability * (1 - reliability) / iterations),
    meets_guarantee = reliability >= R_c,
    wl_mean = mean(results$workload_saved, na.rm = TRUE),
    wl_se = .se(results$workload_saved, iterations),
    need_see_mean = mean(results$pct_needed_to_find_target, na.rm = TRUE),
    need_see_se = .se(results$pct_needed_to_find_target, iterations),
    keep_pct = keep_pct,
    n_kept = n_keep
  )

  list(summary = summary, iterations = results, best_seeds = best_seeds, best = best)
}

