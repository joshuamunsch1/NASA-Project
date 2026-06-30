# ============================================================
# countNormalize_new_studies.R  
#
# New single-study DESeq handlers for three OSDR datasets.
# Assumed pairing (count file  <->  metadata .txt):
#   OSD-270 : GLDS-270_rna_seq_RSEM_Unnormalized_Counts.csv            <-> 270.txt
#   OSD-580 : GLDS-573_rna_seq_RSEM_Unnormalized_Counts.csv            <-> 580.txt
#   OSD-599 : GLDS-596_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv <-> 599.txt
# ============================================================


# ---- setup (copied from countNormalize.R) ------------------
library(DESeq2)
library(dplyr)

setwd("C:/Users/Joshua/Desktop/Project")  # match countNormalize.R

# defaults equivalent to the optparse options in countNormalize.R.
# (no command-line parsing needed when sourcing interactively)
opt <- list(
  feature_list = "prefiltered_pseudogenes.csv",
  metadata_dir = "single_study_metadata",
  counts_dir   = "raw_counts"
)

# subset the counts matrix to the provided feature list (ENSEMBL IDs)
parse_features <- function(counts) {
  if (!is.null(opt$feature_list)) {
    features <- read.table(opt$feature_list, header = TRUE, sep = ",")
    counts <- counts[row.names(counts) %in% features[, 2], ]
  }
  counts
}

file_imports <- function(counts_fname, metadata_fname) {
  counts_path <- paste(getwd(), "data", "raw_counts",
                       paste0(counts_fname, ".csv"), sep = "/")
  metadata_path <- paste(getwd(), "data", "single_study_metadata",
                         paste0(metadata_fname, ".txt"), sep = "/")
  counts  <- read.delim(counts_path, sep = ",", row.names = 1, header = TRUE)
  coldata <- read.delim(metadata_path, row.names = 1, header = TRUE, sep = "\t")
  list(counts = counts, coldata = coldata)
}

drop_ercc <- function(counts) {
  ercc <- grep("^ERCC", rownames(counts))
  if (length(ercc) > 0) counts <- counts[-ercc, ]
  counts
}

# arms that are never part of the SF-vs-GC contrast.
# both "Vivarium_Control" and "Vivarium_control" are listed because
# OSD-580 publishes the value lower-cased ("Vivarium control").
NON_FLIGHT_ARMS <- c("Basal_Control", "Vivarium_Control", "Vivarium_control")

glds_sfgc_dds <- function(counts_fname, metadata_fname, out_csv,
                          match_col = "Sample.Name",
                          drop_ercc_spikes = TRUE,
                          ref_level = "Ground_Control") {
  
  data    <- file_imports(counts_fname, metadata_fname)  # row.names = 1 = Source Name
  counts  <- data$counts
  coldata <- data$coldata
  
  if (drop_ercc_spikes) counts <- drop_ercc(counts)
  counts <- parse_features(counts)
  
  coldata$spaceflight <- coldata$Factor.Value.Spaceflight. %>%
    { gsub(" ", "_", .) } %>%   # nolint
    { gsub("#", "", .) }        # nolint
  
  coldata <- coldata[!coldata$spaceflight %in% NON_FLIGHT_ARMS, ]
  #  coldata = coldata[!grepl("LAR", rownames(coldata)),]
  
  coldata$key <- coldata[[match_col]] %>% { gsub("-", ".", .) } %>% factor()
  
  counts <- counts[, colnames(counts) %in% coldata$key]
  write.csv(counts, file = out_csv)
  coldata <- coldata[match(colnames(counts), coldata$key), ]
  
  coldata$spaceflight <- factor(coldata$spaceflight) %>% relevel(ref = ref_level)
  
  cat("Dimensions of Counts:", dim(counts), "\n")
  cat("Dimensions of Metadata:", dim(coldata), "\n")
  
  row.names(coldata) <- coldata$key
  dds_nofilt <- DESeqDataSetFromMatrix(countData = round(counts),
                                       colData   = coldata,
                                       design    = ~ spaceflight)
  
  list(data = dds_nofilt, metadata = coldata)
}

# ------------------------------------------------------------
# OSD-580  (GLDS-573)  -- 4 arms in metadata; keep SF vs GC.
# ------------------------------------------------------------
glds580_dds <- function(counts_fname, metadata_fname) {
  glds_sfgc_dds(counts_fname, metadata_fname, out_csv = "580.csv",
                match_col = "Sample.Name", drop_ercc_spikes = TRUE)
}

# ------------------------------------------------------------
# OSD-599  (GLDS-596)  -- only GC + SF present.
#   Verified: the GLDS-596 count columns ARE the GSM accessions
#   (GSM6996077..82), so the Sample Name column matches directly.
# ------------------------------------------------------------
glds599_dds <- function(counts_fname, metadata_fname) {
  glds_sfgc_dds(counts_fname, metadata_fname, out_csv = "599.csv",
                match_col = "Sample.Name", drop_ercc_spikes = TRUE)
}

# ------------------------------------------------------------
# OSD-270  (GLDS-270)  -- bulk heart RSEM matrix: 5 samples,
#   3 SF (RR3_HRT_FLT_F1/F2/F9) + 2 GC (RR3_HRT_GC_G7/G8).
#   270.txt was rebuilt to key on these column names, with a
#   unique Source Name per sample, so it is a plain SF/GC study.
#   (The original 46-row s_OSD-270 sample table was a different,
#   sectioned assay and is NOT used here.)
# ------------------------------------------------------------
glds270_dds <- function(counts_fname, metadata_fname) {
  glds_sfgc_dds(counts_fname, metadata_fname, out_csv = "270.csv",
                match_col = "Sample.Name", drop_ercc_spikes = TRUE)
}

# ============================================================
# RUNNER BLOCK  
# ============================================================
glds270_output <- glds270_dds("GLDS-270_rna_seq_RSEM_Unnormalized_Counts", "270")
glds580_output <- glds580_dds("GLDS-573_rna_seq_RSEM_Unnormalized_Counts", "580")
glds599_output <- glds599_dds("GLDS-596_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq", "599")

dds270 <- estimateSizeFactors(glds270_output$data)
dds580 <- estimateSizeFactors(glds580_output$data)
dds599 <- estimateSizeFactors(glds599_output$data)

norm_270 <- counts(dds270, normalized = TRUE)
norm_580 <- counts(dds580, normalized = TRUE)
norm_599 <- counts(dds599, normalized = TRUE)

colnames(norm_270) <- gsub("\\.", "-", colnames(norm_270))
colnames(norm_580) <- gsub("\\.", "-", colnames(norm_580))
colnames(norm_599) <- gsub("\\.", "-", colnames(norm_599))

norm_path <- paste(getwd(), "data", "norm_counts", sep = "/")
write.table(round(norm_270), file = paste0(norm_path, "/", "270.csv"),
            sep = ",", row.names = TRUE, col.names = NA)
write.table(round(norm_580), file = paste0(norm_path, "/", "580.csv"),
            sep = ",", row.names = TRUE, col.names = NA)
write.table(round(norm_599), file = paste0(norm_path, "/", "599.csv"),
            sep = ",", row.names = TRUE, col.names = NA)