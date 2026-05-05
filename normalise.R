
library(DESeq2)
library(dplyr)
library(optparse)
setwd('C:/Users/Joshua/Downloads')
option_list <- list(
  make_option(c("-F", "--feature_list"),
              type    = "character",
              default = 'prefiltered_pseudogenes.csv',
              help    = "gene set filtering file name (CSV, gene IDs in 2nd column)",
              metavar = "character"),
  make_option(c("-M", "--metadata_file"),
              type    = "character",
              default = "meta.csv",          # CHANGED: single file, not a directory
              help    = "combined metadata CSV file name",
              metavar = "character"),
  make_option(c("-C", "--counts_file"),
              type    = "character",
              default = "counts_complete_corr.csv",  # CHANGED: single combined counts file
              help    = "combined raw counts CSV file name",
              metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

if (!is.null(opt$feature_list)) {
  print(paste("Feature list:", opt$feature_list,
              "| Metadata:", opt$metadata_file,
              "| Counts:", opt$counts_file))
}

parse_features <- function(counts) {
  if (!is.null(opt$feature_list)) {
    cat('parsing files')
    features <- read.table(opt$feature_list, header = TRUE, sep = ",")
    print(paste(features[1, 2]))
    counts <- counts[row.names(counts) %in% features$X, ]
  }
  return(counts)
}


load_study <- function(counts_all, metadata_all, study_id) {
  coldata <- metadata_all[metadata_all$study == study_id, ]
  counts  <- counts_all[, coldata$sample, drop = FALSE]
  return(list(counts = counts, coldata = coldata))
}

# ── Load combined inputs ──────────────────────────────────────────────────────
counts_all  <- read.delim(opt$counts_file,
                          sep       = ",",
                          row.names = 1,
                          header    = TRUE,
                          check.names = FALSE)

metadata_all <- read.delim(opt$metadata_file,
                           sep    = ",",
                           header = TRUE)

cat("Loaded combined counts:", nrow(counts_all), "genes x",
    ncol(counts_all), "samples\n")
cat("Studies in metadata:", paste(sort(unique(metadata_all$study)), collapse = ", "), "\n")


study_a_dds <- function() {
  data    <- load_study(counts_all, metadata_all, "A")
  counts  <- data$counts
  coldata <- data$coldata
  
  counts <- parse_features(counts)
  write.csv(counts, 'A.csv')
  

  coldata$spaceflight <- factor(
    ifelse(coldata$group == 0, "Spaceflight", "Ground_Control"),
    levels = c("Ground_Control", "Spaceflight")   # GC as reference, same as original
  )
  
  row.names(coldata) <- coldata$sample
  
  cat("Dimensions of Study A Counts:", dim(counts), "\n")
  cat("Dimensions of Study A Metadata:", dim(coldata), "\n")
  
  # Design formula — identical to original simple studies (glds47, glds242)
  factor_names   <- c("spaceflight")
  design_formula <- paste("", paste(factor_names, collapse = " + "), sep = " ~ ")
  cat("The design formula is:", design_formula, "\n")
  
  dds_nofilt <- DESeqDataSetFromMatrix(
    countData = round(counts),   # round() identical to original
    colData   = coldata,
    design    = as.formula(design_formula)
  )
  
  return(list(data = dds_nofilt, metadata = coldata))
}


study_b_dds <- function() {
  data    <- load_study(counts_all, metadata_all, "B")
  counts  <- data$counts
  coldata <- data$coldata
  
  counts <- parse_features(counts)
  write.csv(counts, 'B.csv')
  
  coldata$spaceflight <- factor(
    ifelse(coldata$group == 0, "Spaceflight", "Ground_Control"),
    levels = c("Ground_Control", "Spaceflight")
  )
  
  row.names(coldata) <- coldata$sample
  
  cat("Dimensions of Study B Counts:", dim(counts), "\n")
  cat("Dimensions of Study B Metadata:", dim(coldata), "\n")
  
  factor_names   <- c("spaceflight")
  design_formula <- paste("", paste(factor_names, collapse = " + "), sep = " ~ ")
  cat("The design formula is:", design_formula, "\n")
  
  dds_nofilt <- DESeqDataSetFromMatrix(
    countData = round(counts),
    colData   = coldata,
    design    = as.formula(design_formula)
  )
  
  return(list(data = dds_nofilt, metadata = coldata))
}

study_c_dds <- function() {
  data    <- load_study(counts_all, metadata_all, "C")
  counts  <- data$counts
  coldata <- data$coldata
  
  counts <- parse_features(counts)
  write.csv(counts, 'C.csv')
  
  coldata$spaceflight <- factor(
    ifelse(coldata$group == 0, "Spaceflight", "Ground_Control"),
    levels = c("Ground_Control", "Spaceflight")
  )
  coldata$age <- factor(coldata$age)
  coldata$euthanasia <- factor(coldata$euthanasia)
  
  row.names(coldata) <- coldata$sample
  
  cat("Dimensions of Study C Counts:", dim(counts), "\n")
  cat("Dimensions of Study C Metadata:", dim(coldata), "\n")
  
 # factor_names   <- c("spaceflight")
  factor_names   <- c("spaceflight", "age")
  design_formula <- paste("", paste(factor_names, collapse = " + "), sep = " ~ ")
  cat("The design formula is:", design_formula, "\n")
  
  dds_nofilt <- DESeqDataSetFromMatrix(
    countData = round(counts),
    colData   = coldata,
    design    = as.formula(design_formula)
  )
  
  return(list(data = dds_nofilt, metadata = coldata))
}

study_a_output <- study_a_dds()
study_b_output <- study_b_dds()
study_c_output <- study_c_dds()

study_a_dds_obj <- estimateSizeFactors(study_a_output$data)
study_b_dds_obj <- estimateSizeFactors(study_b_output$data)
study_c_dds_obj <- estimateSizeFactors(study_c_output$data)

norm_A <- counts(study_a_dds_obj, normalized = TRUE)
norm_B <- counts(study_b_dds_obj, normalized = TRUE)
norm_C <- counts(study_c_dds_obj, normalized = TRUE)
 

write.table(round(norm_A),
            file      = file.path("C:/Users/Joshua/Downloads/data/norm_counts/", "A.csv"),
            sep       = ",",
            row.names = TRUE,
            col.names = NA)

write.table(round(norm_B),
            file      = file.path("C:/Users/Joshua/Downloads/data/norm_counts/", "B.csv"),
            sep       = ",",
            row.names = TRUE,
            col.names = NA)

write.table(round(norm_C),
            file      = file.path("C:/Users/Joshua/Downloads/data/norm_counts/", "C.csv"),
            sep       = ",",
            row.names = TRUE,
            col.names = NA)

counts <- read.csv('C:/Users/Joshua/Downloads/data/norm_counts/A.csv')
row.names(counts) <- counts$X
counts$X <- NULL
write.csv(counts, 'C:/Users/Joshua/Downloads/data/norm_counts/A.csv')

counts <- read.csv('C:/Users/Joshua/Downloads/data/norm_counts/B.csv')
row.names(counts) <- counts$X
counts$X <- NULL
write.csv(counts, 'C:/Users/Joshua/Downloads/data/norm_counts/B.csv')

counts <- read.csv('C:/Users/Joshua/Downloads/data/norm_counts/C.csv')
row.names(counts) <- counts$X
counts$X <- NULL
write.csv(counts, 'C:/Users/Joshua/Downloads/data/norm_counts/C.csv')
