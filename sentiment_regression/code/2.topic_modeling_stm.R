# ------------------------------------------------------------------------
# STM Topic Modeling for Social Identity Bias Analysis  
# Based on si-study-i-stm.R implementation
# Simplified version without visualization and extra analysis
# ------------------------------------------------------------------------

# Load required libraries
suppressPackageStartupMessages({
  library(stm)
  library(dplyr)
  library(readr)
  library(jiebaR)  # Chinese word segmentation
})

# Initialize jieba
jieba <- worker()

# Load stopwords from file (following 3_segmentation.py approach)
load_stopwords <- function(stopword_file = "../../bias_in_llm/data/hit_stopwords.txt") {
  if (file.exists(stopword_file)) {
    cat(sprintf("Loading stopwords from: %s\n", stopword_file))
    stopwords <- readLines(stopword_file, encoding = "UTF-8", warn = FALSE)
    stopwords <- trimws(stopwords)  # Remove whitespace
    stopwords <- stopwords[nchar(stopwords) > 0]  # Remove empty lines
    cat(sprintf("Loaded %d stopwords\n", length(stopwords)))
    return(stopwords)
  } else {
    cat(sprintf("Warning: Stopword file not found: %s\n", stopword_file))
    cat("Using default Chinese stopwords\n")
    return(c("我们", "他们", "是", "的", "了", "在", "有", "和", "就", "都", "会", "说", "要"))
  }
}

# Function to perform Chinese word segmentation with stopword removal
segment_chinese_text <- function(texts, stopwords = NULL) {
  cat("Performing Chinese word segmentation...\n")
  
  # Load stopwords if not provided
  if (is.null(stopwords)) {
    stopwords <- load_stopwords()
  }
  
  # Function to segment single text
  segment_single <- function(text) {
    if (is.na(text) || nchar(text) == 0) {
      return("")
    }
    
    # Keep only Chinese characters (following 3_segmentation.py approach)
    text_chinese <- gsub("[^\u4e00-\u9fa5]", "", text)
    
    # Segment text using jieba
    words <- segment(text_chinese, jieba)
    
    # Filter words: remove stopwords and keep meaningful words
    valid_words <- words[
      nchar(words) > 0 &  # Non-empty
      !words %in% stopwords &  # Not in stopword list
      nchar(trimws(words)) > 0  # Not just whitespace
    ]
    
    # Return space-separated words
    return(paste(valid_words, collapse = " "))
  }
  
  # Apply segmentation to all texts
  segmented_texts <- sapply(texts, segment_single, USE.NAMES = FALSE)
  
  cat(sprintf("Segmentation completed. %d texts processed.\n", length(segmented_texts)))
  
  return(segmented_texts)
}

# Function to extract dataset name from file path
extract_dataset_name <- function(file_path) {
  filename <- basename(file_path)
  filename_no_ext <- tools::file_path_sans_ext(filename)
  
  if (grepl("^1\\.group_data_", filename_no_ext)) {
    dataset_name <- sub("^1\\.group_data_", "", filename_no_ext)
  } else {
    dataset_name <- "default"
    cat("Warning: Could not extract dataset name from filename, using 'default'\n")
  }
  
  return(dataset_name)
}

# Main STM modeling function
run_stm_modeling <- function(input_file = "./result/1.group_data_WildChat-1M.csv", 
                           K = 60, 
                           stopword_file = "../../bias_in_llm/data/hit_stopwords.txt",
                           search_optimal_K = TRUE,
                           K_range = c(20, 40, 60, 80)) {
  
  cat("=== STM Topic Modeling ===\n")
  
  # Extract dataset name for output files
  dataset_name <- extract_dataset_name(input_file)
  cat(sprintf("Dataset: %s\n", dataset_name))
  
  # Load data
  cat("Loading data...\n")
  if (!file.exists(input_file)) {
    stop(sprintf("Error: File not found: %s", input_file))
  }
  
  alldata <- read_csv(input_file, locale = locale(encoding = "UTF-8"))
  cat(sprintf("Data loaded. Total records: %d\n", nrow(alldata)))
  
  # Scale total_tokens if available
  if ("total_tokens" %in% names(alldata)) {
    alldata$total_tokens_scaled <- as.numeric(scale(alldata$total_tokens))
  }
  
  # Extract text data
  df_text <- alldata$text
  
  # Step 1: Chinese word segmentation with stopword removal
  cat("Step 1: Chinese word segmentation...\n")
  stopwords <- load_stopwords(stopword_file)
  segmented_texts <- segment_chinese_text(df_text, stopwords)
  
  # Remove empty segmented texts
  valid_indices <- which(!is.na(segmented_texts) & nchar(segmented_texts) > 0)
  segmented_texts_valid <- segmented_texts[valid_indices]
  alldata_valid <- alldata[valid_indices, ]
  
  cat(sprintf("Valid texts after segmentation: %d (removed %d empty texts)\n", 
              length(segmented_texts_valid), 
              length(df_text) - length(segmented_texts_valid)))
  
  # Step 2: STM text processing on segmented texts
  cat("Step 2: STM text processing on segmented texts...\n")
  
  # Text preprocessing - optimized for segmented Chinese text
  # Note: stopwords already removed during segmentation step
  processed <- textProcessor(segmented_texts_valid,
    metadata = alldata_valid,
    lowercase = TRUE,
    removestopwords = TRUE,  # Already handled in segmentation
    removenumbers = TRUE,
    removepunctuation = TRUE,
    ucp = FALSE,
    stem = FALSE,  # No stemming for Chinese
    wordLengths = c(1, 15),  # Reasonable word length for Chinese words
    sparselevel = 1,
    verbose = TRUE,
    onlycharacter = FALSE,  # Allow Chinese characters
    striphtml = TRUE,
    custompunctuation = NULL,
    customstopwords = c("我们是", "我们的是", "我们通常", "我们的方式是", "我们经常", "我们相信", "他们是", "他们的是", "他们通常", "他们的方式是", "他们经常", "他们相信")
  )
  
  cat("Step 3: Preparing documents...\n")
  
  # Prepare documents - use lower threshold for Chinese text
  out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
    lower.thresh = 10  # More lenient for Chinese
  )
  
  cat(sprintf("Vocabulary size: %d\n", length(out$vocab)))
  cat(sprintf("Documents after filtering: %d\n", length(out$documents)))
  
  # Step 4: Model Selection (optional)
  if (search_optimal_K) {
    cat("Step 4: Model Selection - Finding optimal K...\n")
    
    # Search for optimal number of topics
    cat(sprintf("Testing different K values: %s\n", paste(K_range, collapse = ", ")))
    cat("This may take several minutes...\n")
    
    # Try K selection with error handling
    tryCatch({
      kResult <- searchK(out$documents, out$vocab,
                         K = K_range,
                         data = out$meta,
                         verbose = TRUE
      )
    }, error = function(e) {
      cat(sprintf("Error in searchK: %s\n", e$message))
      cat(sprintf("Falling back to original K = %d\n", K))
      search_optimal_K <<- FALSE  # Disable K search for this run
      return(NULL)
    })
    
    # Save K selection plot (only if searchK succeeded)
    if (exists("kResult") && !is.null(kResult)) {
      kplot_file <- sprintf("./result/kplot_%s.pdf", dataset_name)
      pdf(kplot_file)
      plot(kResult)
      dev.off()
      cat(sprintf("K selection plot saved to: %s\n", kplot_file))
    } else {
      cat("K selection failed, skipping plot generation.\n")
      search_optimal_K <- FALSE
    }
    
    # Display results and select best K (only if searchK succeeded)
    if (search_optimal_K && exists("kResult") && !is.null(kResult)) {
      cat("\n=== K Selection Results ===\n")
      print(kResult$results)
      
      # Debug: Check the structure of results
      cat("\n=== Debug: Structure of kResult$results ===\n")
      str(kResult$results)
      
      # Auto-select K based on semantic coherence and exclusivity balance
    # Higher semantic coherence and exclusivity are better
    results_df <- kResult$results
    
    # Convert list columns to numeric vectors if necessary
    if (is.list(results_df$semcoh)) {
      results_df$semcoh <- unlist(results_df$semcoh)
    }
    if (is.list(results_df$exclus)) {
      results_df$exclus <- unlist(results_df$exclus)
    }
    if (is.list(results_df$K)) {
      results_df$K <- unlist(results_df$K)
    }
    
    # Check if we have valid numeric values
    if (all(is.na(results_df$semcoh)) || all(is.na(results_df$exclus)) || 
        length(results_df$semcoh) == 0 || length(results_df$exclus) == 0) {
      cat("Warning: Invalid semantic coherence or exclusivity values. Using middle K value.\n")
      optimal_K <- K_range[ceiling(length(K_range)/2)]
    } else {
      # Normalize metrics to 0-1 scale for comparison
      semcoh_range <- max(results_df$semcoh, na.rm = TRUE) - min(results_df$semcoh, na.rm = TRUE)
      exclus_range <- max(results_df$exclus, na.rm = TRUE) - min(results_df$exclus, na.rm = TRUE)
      
      # Avoid division by zero
      if (semcoh_range == 0) {
        results_df$semcoh_norm <- rep(0.5, length(results_df$semcoh))
      } else {
        results_df$semcoh_norm <- (results_df$semcoh - min(results_df$semcoh, na.rm = TRUE)) / semcoh_range
      }
      
      if (exclus_range == 0) {
        results_df$exclus_norm <- rep(0.5, length(results_df$exclus))
      } else {
        results_df$exclus_norm <- (results_df$exclus - min(results_df$exclus, na.rm = TRUE)) / exclus_range
      }
      
      # Combined score (you can adjust weights as needed)
      results_df$combined_score <- 0.5 * results_df$semcoh_norm + 0.5 * results_df$exclus_norm
      
      # Select K with highest combined score
      optimal_K <- results_df$K[which.max(results_df$combined_score)]
    }
    
    cat(sprintf("Auto-selected optimal K: %d (based on semantic coherence + exclusivity)\n", optimal_K))
    
    # Use the optimal K instead of the input K
    K <- optimal_K
    
      # Save K selection results
      k_results_file <- sprintf("./result/k_selection_results_%s.csv", dataset_name)
      write_csv(results_df, k_results_file)
      cat(sprintf("K selection results saved to: %s\n", k_results_file))
      
    } else {
      cat("K selection process failed or was disabled. Using original K value.\n")
      # Keep the original K value
    }
    
  } else {
    cat(sprintf("Step 4: Using predefined K = %d (skipping model selection)\n", K))
  }
  
  # Step 5: Train STM model
  if (search_optimal_K) {
    cat(sprintf("Step 5: Training STM model with optimal K=%d topics...\n", K))
  } else {
    cat(sprintf("Step 5: Training STM model with K=%d topics...\n", K))
  }
  
  set.seed(17)  # Same seed as original
  stm_model <- stm(out$documents, out$vocab,
    K = K,
    data = out$meta,
    init.type = "Spectral",
    seed = 17
  )
  
  cat("STM modeling completed.\n")
  
  # Extract dominant topics
  dominant_topics <- max.col(as.data.frame(stm_model$theta))
  
  # Step 6: Map results back to original data
  cat("Step 6: Mapping results back to original data...\n")
  
  # Initialize all records with NA
  alldata$stm_topic <- NA
  alldata$stm_topic_probability <- NA
  
  # Map results back through the filtering steps
  # First map to valid segmented texts
  final_indices <- valid_indices[1:length(dominant_topics)]  # Only those that survived all filtering
  
  # Add topic information
  alldata$stm_topic[final_indices] <- dominant_topics
  
  # Add topic probabilities
  theta_max <- apply(stm_model$theta, 1, max)
  alldata$stm_topic_probability[final_indices] <- theta_max
  
  # Step 7: Save results
  output_file <- sprintf("./result/2.topic_data_stm_%s.csv", dataset_name)
  valid_data <- alldata[!is.na(alldata$stm_topic), ]
  
  # Clean data structure before saving - ensure no list or matrix columns
  cat("Cleaning data structure for CSV export...\n")
  
  # Check and fix column types
  for (i in 1:ncol(valid_data)) {
    col_class <- class(valid_data[[i]])
    if (any(c("list", "matrix", "array") %in% col_class)) {
      cat(sprintf("Warning: Converting column %d (%s) from %s to character\n", 
                  i, names(valid_data)[i], paste(col_class, collapse = ", ")))
      valid_data[[i]] <- as.character(valid_data[[i]])
    }
  }
  
  # Ensure all columns are basic types
  write_csv(valid_data, output_file)
  
  cat(sprintf("Results saved to: %s\n", output_file))
  cat(sprintf("Valid records: %d out of %d original records\n", nrow(valid_data), nrow(alldata)))
  cat(sprintf("Success rate: %.1f%%\n", (nrow(valid_data)/nrow(alldata))*100))
  
  # Save STM objects for potential future use
  stm_file <- sprintf("./result/stm_model_%s.RData", dataset_name)
  save(stm_model, out, dominant_topics, file = stm_file)
  cat(sprintf("STM model saved to: %s\n", stm_file))
  
  return(list(
    stm_model = stm_model,
    out = out,
    enhanced_data = valid_data,
    dominant_topics = dominant_topics
  ))
}

# Run modeling if script is executed directly
if (!interactive()) {
  # Default execution with K search enabled
  results <- run_stm_modeling(
    input_file = "./result/1.group_data_WildChat-1M.csv", 
    K = 60,  # This will be overridden by optimal K if search_optimal_K = TRUE
    stopword_file = "../../bias_in_llm/data/hit_stopwords.txt",
    search_optimal_K = TRUE,
    K_range = c(20, 40, 60, 80)
  )
} 