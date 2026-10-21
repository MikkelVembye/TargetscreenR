# Function to load embeddings from a file, using a cache to avoid reloading if already loaded.
# We use local here to create a closure that holds the cache environment, so each worker has its own cache.
# Meaning that each worker will load the embeddings once per (dataset, model) combination, and reuse them for all iterations.
.load_embeddings <- local({

  cache <- new.env(parent = emptyenv())

  function(data_name, model, dir) {
    
    # Use a unique key for the cache based on dataset name and model
    key <- paste0(data_name, "_", model)

    # If the embeddings for this (dataset, model) combination are already in the cache, return them
    if (identical(cache$key, key)) return(cache$value)

    # If not, load the embeddings from the file and store them in the cache
    # (works for sentence-transformers models and for "tfidf"); same file naming as in embed_corpus()
    model_file <- gsub("/", "--", model, fixed = TRUE)
    path <- file.path(dir, paste0(data_name, "_", model_file, ".rds"))
    if (!file.exists(path)) {
      stop("No embedding file for model '", model, "' at ", path, ". Run embed_corpus() first.", call. = FALSE)
    }

    # Drop the previous matrix before reading the next one so the two never coexist
    cache$key   <- NULL
    cache$value <- NULL
    # Force garbage collection to free memory before loading the new embeddings
    gc(verbose = FALSE)

    # Read the embeddings from the file and store them in the cache
    cache$value <- readRDS(path)
    cache$key   <- key
    cache$value
  }
})

# Run one iteration of a design with one seed. `design` is a list with the arguments of
# .generate_prioritized_data() that stay the same in every iteration (everything except `seed`).
# Messages and warnings are muted, as they would repeat in every iteration.
.run_iteration <- function(iteration_seed, design) {
  do.call(.generate_prioritized_data, c(design, list(seed = iteration_seed))) |>
    suppressWarnings() |>
    suppressMessages()
}

# The numbers kept for each iteration, taken from the prioritized data of one run
.summarise_run <- function(res) {
  last_target_row <- max(res$row_number[res$is_target == 1L], na.rm = TRUE) # last row of the target
  n_relevant <- sum(res$included_final == 1, na.rm = TRUE) # number of relevant records in the data
  found <- sum(res$included_final == 1 & res$row_number <= last_target_row, na.rm = TRUE) # number of relevant records found before the last target row
  list(
    workload_saved = attr(res, "screen_decisions_info")$workload_saved,
    recall_pstar = if (n_relevant == 0L) NA_real_ else found / n_relevant,
    pct_needed_to_find_target = last_target_row / nrow(res)
  )
}

# Standard error of the mean over n iterations
.se <- function(x, n) {
  if (n > 1) stats::sd(x, na.rm = TRUE) / sqrt(n) else NA_real_
}


# Find an installed Python that has sentence-transformers
.find_python <- function() {
  candidates <- Sys.getenv("RETICULATE_PYTHON")

  conda_envs <- tryCatch(reticulate::conda_list()$python, error = function(e) NULL)
  candidates <- c(candidates, conda_envs)

  # Look in conda install folders
  roots <- unlist(lapply(
    c(Sys.getenv("LOCALAPPDATA"), Sys.getenv("USERPROFILE"), path.expand("~")),
    function(base) file.path(base, c("miniconda3", "anaconda3", "Miniconda3", "Anaconda3"))
  ))
  candidates <- c(
    candidates,
    file.path(roots, "python.exe"), file.path(roots, "bin", "python"),
    Sys.glob(file.path(roots, "envs", "*", "python.exe")),
    Sys.glob(file.path(roots, "envs", "*", "bin", "python"))
  )

  candidates <- unique(candidates[nzchar(candidates) & file.exists(candidates)])

  for (path in candidates) {
    status <- tryCatch(
      suppressWarnings(system2(path, c("-c", shQuote("import sentence_transformers")), stdout = FALSE, stderr = FALSE)),
      error = function(e) 1L
    )
    if (identical(as.integer(status), 0L)) return(normalizePath(path, winslash = "/"))
  }

  NULL
}
