# ============================================================
# 10_dge_singlestudy.R  (per-study DESeq2 DGE -> ranked log2FC)
#
#   inputs : data/raw_counts/<id>.csv   (raw OSDR counts, genes x samples)
#            data/single_study_metadata/<id>.txt  (+ specialty/ for 168, 245)
#            prefiltered_pseudogenes.csv (the same -F feature list)
#   outputs: data/expression_results/glds<ID>_fc_and_pvals.csv
#            columns: gene, log2fc, baseMean, lfcSE, stat, pvalue, padj
# adapted from Ilangovan et al. for benchmarking
# ============================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(dplyr)
})

setwd("path/to/liver") 

# ---- config -------------------------------------------------
FEATURE_LIST <- "prefiltered_pseudogenes.csv"  # same file you pass to -F
OUT_DIR      <- "data/single_dge"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Spaceflight level vs the set of levels pooled into ground control.
# Cohort controls are KEPT as part of GC (matching the study); only
# Vivarium/Basal are excluded (already dropped by the builders).
SF_LEVEL  <- "Space_Flight"
GC_LEVELS <- c("Ground_Control", "Cohort_Control_1", "Cohort_Control_2")
GC_LEVELS_242 <- c("Cohort_Control_1")

# ------------------------------------------------------------
# SHARED HELPERS (identical logic to countNormalize.R)
# ------------------------------------------------------------
parse_features <- function(counts) {
  if (!is.null(FEATURE_LIST) && file.exists(FEATURE_LIST)) {
    features <- read.table(FEATURE_LIST, header = TRUE, sep = ",")
    counts <- counts[row.names(counts) %in% features[, 2], ]
  } else {
    warning("Feature list '", FEATURE_LIST, "' not found - DGE will run on ALL genes.")
  }
  counts
}

file_imports <- function(counts_fname, metadata_fname) {
  counts_path   <- paste(getwd(), "data", "raw_counts",
                         paste0(counts_fname, ".csv"), sep = "/")
  metadata_path <- paste(getwd(), "data", "single_study_metadata",
                         paste0(metadata_fname, ".txt"), sep = "/")
  counts  <- read.delim(counts_path,  sep = ",", row.names = 1, header = TRUE)
  coldata <- read.delim(metadata_path, row.names = 1, header = TRUE, sep = "\t")
  list(counts = counts, coldata = coldata)
}

# ------------------------------------------------------------
# PER-STUDY DESeqDataSet BUILDERS (from countNormalize.R)
# Each returns list(data = <DESeqDataSet>, metadata = <coldata>).
# ------------------------------------------------------------
glds47_dds <- function(counts_fname, metadata_fname) {
  data    <- file_imports(counts_fname, metadata_fname)
  counts  <- data$counts
  coldata <- data$coldata
  counts  <- parse_features(counts)
  
  coldata$spaceflight <- coldata$Factor.Value.Spaceflight. %>%
    { gsub(" ", "_", .) } %>% { gsub("#", "", .) }
  coldata <- coldata[!coldata$spaceflight %in% "Basal_Control", ]
  coldata$Sample.Name <- coldata$Sample.Name %>% { gsub("-", ".", .) } %>% factor()
  
  counts  <- counts[, colnames(counts) %in% coldata$Sample.Name]
  coldata <- coldata[match(colnames(counts), coldata$Sample.Name), ]
  coldata$spaceflight <- factor(coldata$spaceflight)
  row.names(coldata)  <- coldata$Sample.Name
  
  dds <- DESeqDataSetFromMatrix(countData = round(counts), colData = coldata,
                                design = ~ spaceflight)
  list(data = dds, metadata = coldata)
}

glds242_dds <- function(counts_fname, metadata_fname) {
  data    <- file_imports(counts_fname, metadata_fname)
  counts  <- data$counts
  coldata <- data$coldata
  
  ercc <- grep("^ERCC", row.names(counts))
  if (length(ercc) > 0) counts <- counts[-ercc, ]
  counts <- parse_features(counts)
  
  coldata$spaceflight <- coldata$Factor.Value.Spaceflight. %>%
    { gsub(" ", "_", .) } %>% { gsub("#", "", .) }
  coldata <- coldata[!coldata$spaceflight %in% c("Basal_Control", "Vivarium_Control"), ]
  coldata$Sample.Name <- coldata$Sample.Name %>% { gsub("-", ".", .) } %>% factor()
  
  counts  <- counts[, colnames(counts) %in% coldata$Sample.Name]
  coldata <- coldata[match(coldata$Sample.Name, colnames(counts)), ]
  coldata$spaceflight <- factor(coldata$spaceflight)
  row.names(coldata)  <- coldata$Sample.Name
  
  dds <- DESeqDataSetFromMatrix(countData = round(counts), colData = coldata,
                                design = ~ spaceflight)
  list(data = dds, metadata = coldata)
}

glds245_dds <- function(counts_fname, metadata_fname) {
  data    <- file_imports(counts_fname, metadata_fname)
  counts  <- data$counts
  coldata <- data$coldata
  
  extract_path <- paste(getwd(), "data", "single_study_metadata", "specialty",
                        paste0("245_RR6_extraction dates", ".txt"), sep = "/")
  extractdata  <- read.delim(extract_path, row.names = 1, header = TRUE, sep = "\t")
  
  ercc <- grep("^ERCC", row.names(counts))
  if (length(ercc) > 0) counts <- counts[-ercc, ]
  counts <- parse_features(counts)
  
  coldata$spaceflight <- coldata$Factor.Value.Spaceflight. %>%
    { gsub(" ", "_", .) } %>% { gsub("#", "", .) }
  coldata$duration   <- coldata$Factor.Value.Duration.   %>% { gsub("~", "", .) }
  coldata$euthanasia <- coldata$Factor.Value.Euthanasia. %>% { gsub(" ", "_", .) }
  coldata$dissection <- coldata$Factor.Value.Dissection.Condition. %>% { gsub(" ", "_", .) }
  
  coldata  <- coldata[!coldata$spaceflight %in% "Basal_Control", ]
  outliers <- c("LAR Flight 5", "ISS-T Flight 5", "ISS-T Flight 9")
  coldata  <- coldata[!(rownames(coldata) %in% outliers), ]
  
  extractdata <- extractdata[rownames(extractdata) %in% coldata$Sample.Name, ]
  coldata$Sample.Name <- coldata$Sample.Name %>% { gsub("-", ".", .) } %>% factor()
  
  counts  <- counts[, colnames(counts) %in% coldata$Sample.Name]
  coldata <- coldata[match(coldata$Sample.Name, colnames(counts)), ]
  coldata$month_year <- extractdata$Month.year
  
  coldata$spaceflight <- factor(coldata$spaceflight)
  coldata$duration    <- factor(coldata$duration)
  coldata$euthanasia  <- factor(coldata$euthanasia)
  coldata$month_year  <- factor(coldata$month_year)
  row.names(coldata)  <- coldata$Sample.Name
  
  # GLDS-245 keeps batch covariates (your choice).
  dds <- DESeqDataSetFromMatrix(countData = round(counts), colData = coldata,
                                design = ~ spaceflight + duration + euthanasia + month_year)
  list(data = dds, metadata = coldata)
}

.glds168_dds <- function(counts_fname, metadata_fname, mission) {
  counts_path   <- paste(getwd(), "data", "raw_counts",
                         paste0(counts_fname, ".csv"), sep = "/")
  metadata_path <- paste(getwd(), "data", "single_study_metadata", "specialty",
                         paste0(metadata_fname, ".txt"), sep = "/")
  counts  <- read.delim(counts_path, sep = ",", row.names = 1, header = TRUE)
  coldata <- read.delim(metadata_path, row.names = 2, header = TRUE, sep = "\t")
  
  ercc <- grep("^ERCC", row.names(counts))
  if (length(ercc) > 0) counts <- counts[-ercc, ]
  counts <- parse_features(counts)
  
  coldata$spaceflight <- coldata$Factor.Value.Spaceflight. %>%
    { gsub(" ", "_", .) } %>% { gsub("#", "", .) }
  coldata$mission <- coldata$Factor.Value.Space.Mission. %>% { gsub(" ", "_", .) }
  coldata$spikein <- coldata$Factor.Value.Spike.in.Quality.Control. %>% { gsub(" ", "_", .) }
  coldata <- coldata[!coldata$spaceflight %in% c("Basal_Control", "Vivarium_Control"), ]
  coldata$Sample.Name <- row.names(coldata) %>% { gsub("-", ".", .) } %>% factor()
  
  counts  <- counts[, colnames(counts) %in% coldata$Sample.Name]
  coldata <- coldata[match(coldata$Sample.Name, colnames(counts)), ]
  coldata$spaceflight <- factor(coldata$spaceflight)
  coldata$mission     <- factor(coldata$mission)
  coldata$spikein     <- factor(coldata$spikein)
  
  sub_coldata <- coldata %>% dplyr::filter(mission == !!mission)
  sub_counts  <- counts[, colnames(counts) %in% sub_coldata$Sample.Name]
  sub_coldata <- sub_coldata[match(colnames(sub_counts), sub_coldata$Sample.Name), ]
  
  # RR1 only: drop the noERCC technical-replicate samples
  if (mission == "SpaceX-4_(RR1)") {
    cdrop <- grep("noERCC", colnames(sub_counts))
    rdrop <- grep("noERCC", sub_coldata$Sample.Name)
    if (length(cdrop) > 0) sub_counts  <- sub_counts[, -cdrop]
    if (length(rdrop) > 0) sub_coldata <- sub_coldata[-rdrop, ]
  }
  row.names(sub_coldata) <- sub_coldata$Sample.Name
  
  dds <- DESeqDataSetFromMatrix(countData = round(sub_counts), colData = sub_coldata,
                                design = ~ spaceflight)
  list(data = dds, metadata = sub_coldata)
}
glds168_rr1_dds <- function(counts_fname, metadata_fname)
  .glds168_dds(counts_fname, metadata_fname, "SpaceX-4_(RR1)")
glds168_rr3_dds <- function(counts_fname, metadata_fname)
  .glds168_dds(counts_fname, metadata_fname, "SpaceX-8_(RR3)")

glds379_dds <- function(counts_fname, metadata_fname) {
  data    <- file_imports(counts_fname, metadata_fname)
  counts  <- data$counts
  coldata <- data$coldata
  
  ercc <- grep("^ERCC", row.names(counts))
  if (length(ercc) > 0) counts <- counts[-ercc, ]
  counts <- parse_features(counts)
  
  coldata$spaceflight <- coldata$Factor.Value.Spaceflight. %>%
    { gsub(" ", "_", .) } %>% { gsub("#", "", .) }
  coldata  <- coldata[!coldata$spaceflight %in% c("Vivarium_Control", "Basal_Control"), ]
  outliers <- c('FL-ISS-10','FL-ISS-16','FL-ISS-17','FL-LAR-02','FL-LAR-08',
                'FL-LAR-11','FL-LAR-15','FL-LAR-18','HGC-LAR-03','HGC-LAR-05','HGC-LAR-18')
  coldata  <- coldata[!(rownames(coldata) %in% outliers), ]
  
  coldata$Sample.Name <- coldata$Sample.Name %>% { gsub("-", ".", .) }
  counts  <- counts[, colnames(counts) %in% coldata$Sample.Name]
  coldata <- coldata[match(coldata$Sample.Name, colnames(counts)), ]
  coldata$spaceflight <- factor(coldata$spaceflight)
  row.names(coldata)  <- coldata$Sample.Name
  
  dds <- DESeqDataSetFromMatrix(countData = round(counts), colData = coldata,
                                design = ~ spaceflight)
  list(data = dds, metadata = coldata)
}

# ------------------------------------------------------------
# RUN DGE
# ------------------------------------------------------------
run_dge <- function(study_output, out_csv, gc_levels = GC_LEVELS) {
  dds <- study_output$data
  
  # keep Space_Flight + the ground-like levels for this study; anything
  # else (e.g. stray vivarium/basal, or off-campaign controls) is dropped.
  keep <- as.character(dds$spaceflight) %in% c(SF_LEVEL, gc_levels)
  if (sum(keep) < ncol(dds))
    message(sprintf("  dropped %d sample(s) not in SF/GC sets", ncol(dds) - sum(keep)))
  dds <- dds[, keep]
  
  # POOL cohort controls into ground control -> 2-level condition,
  # overwriting `spaceflight` (the variable the design formula uses).
  pooled <- ifelse(as.character(dds$spaceflight) == SF_LEVEL,
                   "Space_Flight", "Ground_Control")
  dds$spaceflight <- factor(pooled, levels = c("Ground_Control", "Space_Flight"))
  
  # drop now-empty factor levels across the remaining colData factors
  for (cn in colnames(SummarizedExperiment::colData(dds))) {
    col <- SummarizedExperiment::colData(dds)[[cn]]
    if (is.factor(col)) SummarizedExperiment::colData(dds)[[cn]] <- droplevels(col)
  }
  dds$spaceflight <- relevel(dds$spaceflight, ref = "Ground_Control")
  
  dds <- DESeq(dds)
  res <- results(dds, contrast = c("spaceflight", "Space_Flight", "Ground_Control"))
  
  out <- data.frame(
    gene     = rownames(res),
    log2fc   = res$log2FoldChange,
    baseMean = res$baseMean,
    lfcSE    = res$lfcSE,
    stat     = res$stat,
    pvalue   = res$pvalue,
    padj     = res$padj,
    row.names = NULL, stringsAsFactors = FALSE
  )
  out <- out[!is.na(out$log2fc), ]                    # genes GSEA can rank
  
  write.csv(out, file = file.path(OUT_DIR, out_csv), row.names = FALSE)
}

# ------------------------------------------------------------
# DRIVE ALL SIX STUDIES
# ------------------------------------------------------------
cat("GLDS-47  (RR1 CASIS)\n");  run_dge(glds47_dds("47", "47"),            "glds47_fc_and_pvals.csv")
cat("GLDS-168 RR1 (NASA)\n");   run_dge(glds168_rr1_dds("168", "168"),     "glds168rr1_fc_and_pvals.csv")
cat("GLDS-168 RR3\n");          run_dge(glds168_rr3_dds("168", "168"),     "glds168rr3_fc_and_pvals.csv")
cat("GLDS-242 (RR9)\n");        run_dge(glds242_dds("242", "242"),         "glds242_fc_and_pvals.csv", gc_levels = GC_LEVELS_242)
cat("GLDS-245 (RR6)\n");        run_dge(glds245_dds("245", "245"),         "glds245_fc_and_pvals.csv")
cat("GLDS-379 (RR8)\n");        run_dge(glds379_dds("379", "379"),         "glds379_fc_and_pvals.csv")

