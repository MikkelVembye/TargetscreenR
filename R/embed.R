#' @title Embed the corpus using a specified model and save the embeddings to a file
#'
#' @description
#' `embed_corpus` embeds the corpus using a specified model and saves the embeddings to a file.
#' As embeddings are deterministic i.e. doesnt vary across iterations. These can be 
#' precomputed and reused, saving time and computational resources. 
#' See (https://medium.com/@karamvirhapal/encoding-vs-embedding-models-both-output-numbers-different-stories-5c85eced1801)
#' 
#' Either a sentence-transformers model or a text2vec TF-IDF matrix can be used for embedding.
#' The sentence-transformers model is a pre-trained transformer model that generates embeddings for text data. 
#' These can be found at https://huggingface.co/sentence-transformers. The special model name "tfidf" gives a 
#' text2vec TF-IDF matrix instead of a sentence-transformers embedding.
#' The TF-IDF use english stopwords, drops terms in fewer than 5 records and terms in more than 90% of the records, 
#' then keeps the tfidf_max_terms most frequent terms.
#'
#' @param data Data frame containing the corpus to be embedded. It needs a unique record id column, a
#'   title column and an abstract column. By default these are called "eppi_id", "title" and
#'   "abstract". Use `id_col`, `title_col` and `abstract_col` if yours have other names.
#' @param model Character string specifying the model to use for embedding. Can be a sentence-transformers 
#'   model name or "tfidf" for a text2vec TF-IDF matrix.
#' @param python_dir Character string specifying the path to the Python executable. Only needed for sentence-transformers models.
#'   If not provided, the function looks for an installed Python (the one in `RETICULATE_PYTHON` and
#'   the conda environments it can find) that has sentence-transformers. If none is found,
#'   reticulate sets up its own Python.
#' @param dir Character string specifying the directory to save the embeddings. If no directory is provided, 
#'   embeddings will be saved in the current working directory.
#' @param batch_size Integer specifying the number of texts to embed at a time. Lower it if the GPU runs out of memory.
#'   Default is 32. Do not change unless torch.OutOfMemoryError.
#' @param tfidf_max_terms Integer specifying the number of most frequent terms to keep as columns for "tfidf" model.
#'  Default is 1000. Reasonable values for this parameter are between 500 and 2000. 
#'  Lower values reduce memory usage and speed up training, but may lose information.
#' @param id_col Character string with the name of the column holding the unique record id.
#'   Default is "eppi_id".
#' @param title_col Character string with the name of the title column. Default is "title".
#' @param abstract_col Character string with the name of the abstract column. Default is "abstract".
#' @param data_name Character string used as the first part of the file name. Default is the name of
#'   the data frame you pass in. You normally do not need to set this.
#'
#' @return Invisibly, the path to the saved `.rds` file. The file holds a matrix with one row per
#'   record (named by the values of `id_col`) and one column per embedding dimension (or per term for "tfidf").
#'   The file is named `<data name>_<model>.rds`. If it already exists, embedding is skipped
#'   and the existing path is returned.
#'
#' @examples
#' \dontrun{
#' # combined_data needs an id, a title and an abstract column (eppi_id, title and abstract by default)
#'
#' # Other column names
#' embed_corpus(combined_data, model = "tfidf", dir = "embeddings",
#'              id_col = "record_id", title_col = "Title", abstract_col = "Abstract")
#'
#' # TF-IDF: no Python needed
#' embed_corpus(combined_data, model = "tfidf", dir = "embeddings")
#'
#' # Sentence-transformers model: reticulate sets up Python on the first run
#' embed_corpus(combined_data, model = "all-MiniLM-L6-v2", dir = "embeddings")
#'
#' # Use your own Python instead
#' embed_corpus(
#'   combined_data,
#'   model = "all-MiniLM-L6-v2",
#'   python_dir = "C:/path/to/python.exe",
#'   dir = "embeddings"
#' )
#'}
#' @export
embed_corpus <-
  function(
    data,
    model = "all-MiniLM-L6-v2",
    python_dir = NULL,
    dir = NULL,
    batch_size = 32L,
    tfidf_max_terms = 1000,
    id_col = "eppi_id",
    title_col = "title",
    abstract_col = "abstract",
    data_name = deparse(substitute(data))
  ) {

  #####################################################
  # Initial checks
  #####################################################
  required_cols <- c(id_col, title_col, abstract_col)
  if (!all(vapply(required_cols, function(x) is.character(x) && length(x) == 1, logical(1)))) {
    stop("`id_col`, `title_col` and `abstract_col` must each be a single column name.")
  }
  if (!all(required_cols %in% colnames(data))) {
    stop("Data frame must contain columns: ", paste(required_cols, collapse = ", "))
  }

  if (!is.null(dir) && !dir.exists(dir)) {
    dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  } else if (is.null(dir)) {
    # Save in the current working directory if no directory is provided
    warning("No directory provided. Embeddings will be saved in the current working directory.")
    dir <- getwd()
  }

  if (!is.character(model) || length(model) != 1) {
    stop("Model must be a single character string.")
  }

  if (model == "tfidf" && !is.numeric(tfidf_max_terms)) {
    stop("tfidf_max_terms must be a numeric value.")
  }

  if (!is.numeric(batch_size) || batch_size <= 0) {
    stop("Batch size must be a positive numeric value.")
  }

  # data_name (the name of the data frame as a string) is part of the file name
  if (!is.character(data_name) || length(data_name) != 1 || grepl("[^A-Za-z0-9._-]", data_name)) {
    stop("Pass the data frame by its plain variable name (e.g. `combined_data`), not as an expression like `",
         paste(data_name, collapse = ""), "`. The name is used in the file name.", call. = FALSE)
  }
  # HF model ids can contain "/" (e.g. "microsoft/harrier-oss-v1-270m"), which isn't valid in a filename
  model_file <- gsub("/", "--", model, fixed = TRUE)
  path <- file.path(dir, paste0(data_name, "_", model_file, ".rds"))

  if (file.exists(path)) {
    message("Embeddings for model '", model, "' already exist at ", path, ". Skipping embedding.")
    return(invisible(path))
  }
  

  # Every record in the corpus is embedded, so any subset an iteration draws can be looked up
  # by eppi_id. Ids must be unique.
  ids <- as.character(data[[id_col]])
  stopifnot(!anyNA(ids), !anyDuplicated(ids))

  # Embed the corpus by concatenating the title and abstract for each record
  text <- paste(data[[title_col]], data[[abstract_col]])

  if (model == "tfidf") {
    # Stopwords, terms in fewer than 5 records and terms in more than 90% of the records are dropped,
    # then the tfidf_max_terms most frequent terms are kept. Fitted on the whole corpus without labels.

    # Create an iterator. This is used as input to the vocabulary creation and DTM creation functions.
    it <- text2vec::itoken(text, # iterator over the text data
                          preprocessor = tolower, # convert to lowercase
                          tokenizer = text2vec::word_tokenizer, # tokenize into words
                           ids = ids, # unique ids for each record
                           progressbar = TRUE)

    vocab <- text2vec::create_vocabulary(it, stopwords = stopwords::stopwords("en", source = "smart")) |>
      text2vec::prune_vocabulary(term_count_min = 5, # drop terms that appear in fewer than 5 records
                                doc_proportion_max = 0.9, # drop terms that appear in more than 90% of records
                                vocab_term_max = tfidf_max_terms) # keep only the most frequent terms

    dtm <- text2vec::create_dtm(it, text2vec::vocab_vectorizer(vocab)) # create a document-term matrix (DTM) using the pruned vocabulary
    embeddings <- as.matrix(text2vec::TfIdf$new()$fit_transform(dtm)) # compute the TF-IDF weights and convert to a matrix
  } else {
    # Use reticulate to call Python and load the sentence-transformers library
    # Step 1: set up Python and load sentence-transformers
    sentence_transformers <- tryCatch({
      # No python_dir given: look for an installed Python that already has sentence-transformers
      if (is.null(python_dir)) python_dir <- .find_python()

      if (!is.null(python_dir)) {
        message("Using Python at ", python_dir)
        reticulate::use_python(python_dir, required = TRUE)
      } else {
        message("No Python with sentence-transformers found. reticulate will set up its own Python. The first run can take a few minutes.")
        reticulate::py_require("sentence-transformers") # if no python is found, let reticulate download and use a default Python.
      }
      reticulate::import("sentence_transformers")
    }, error = function(e) {
      stop("Could not set up Python or load 'sentence_transformers'.\n",
           "Check your internet connection, or give python_dir a Python that has sentence-transformers installed.\n",
           "Original error: ", conditionMessage(e), call. = FALSE)
    })

    # Step 2: download/load the model
    embed_model <- tryCatch(
      sentence_transformers$SentenceTransformer(model, trust_remote_code = TRUE),
      error = function(e) {
        stop("Could not load model '", model, "'.\n",
             "Check the spelling of the model name and your internet connection.\n",
             "Original error: ", conditionMessage(e), call. = FALSE)
      })

    # Step 3: embed the texts
    embeddings <- tryCatch(
      embed_model$encode(text, batch_size = as.integer(batch_size)),
      error = function(e) {
        stop("Embedding failed.\n",
             "If the error mentions 'out of memory', try a smaller batch_size (e.g. 8).\n",
             "Original error: ", conditionMessage(e), call. = FALSE)
      })
  }

  # ranger's x/y matrix interface requires named columns to recognize covariates.
  # Embedding dimensions have no meaning, so they are just numbered; TF-IDF keeps the words as column names.
  if (model != "tfidf") {
    colnames(embeddings) <- paste0("V", seq_len(ncol(embeddings)))
  }

  rownames(embeddings) <- ids

  # Save the embeddings as parquet
  arrow::write_parquet(embeddings, path)

  invisible(path)
  }