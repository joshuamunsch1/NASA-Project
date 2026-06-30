# ============================================================
# 10_dge_singlestudy.R
# ============================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(dplyr)
})

setwd("path/to/heart")  

# ---- config -------------------------------------------------
FEATURE_LIST <- "prefiltered_pseudogenes.csv"  # same file you pass to -F
OUT_DIR      <- "data/single_dge"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

SF_LEVEL  <- "Space_Flight"
GC_LEVELS <- c("Ground_Control", "Cohort_Control_1", "Cohort_Control_2")



# GLDS-573 (OSD-580, RRRM-2) DESIGN TOGGLE.
# Its 40 SF/GC samples form a balanced 2x2x2 factorial: each arm (FLT, GC)
# has exactly 10 ISS-T / 10 LAR (dissection) and 10 Young / 10 Old (age), so
# neither factor is confounded with spaceflight. Both designs give an unbiased
# SF effect; blocking on age + dissection soaks up that (real, balanced)
# variance and raises power, analogous to the GLDS-245 covariate choice.
#   FALSE -> ~ spaceflight                      (pipeline default; matches 47/270/599)
#   TRUE  -> ~ spaceflight + age + dissection   (blocked / more powerful)
BLOCK_580 <- FALSE

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
# NEW STUDIES (OSD-270 / 580 / 599) -- plain SF-vs-GC builders.
# Identical construction to countNormalize_new_studies.R::glds_sfgc_dds
# (conditional ERCC removal -> feature subset -> drop non-flight arms ->
# match Sample Name), so the gene/sample set matches the normalization
# pipeline. counts_fname is the raw GeneLab basename in data/raw_counts/;
# metadata_fname is the short id (270/580/599) in single_study_metadata/.
#   270 (GLDS-270): 5 bulk-heart samples, 3 SF / 2 GC, no ERCC.
#   580 (GLDS-573): 40 SF/GC after Basal+Vivarium dropped; 92 ERCC rows.
#   599 (GLDS-596): 6 GSM-named samples, 3 SF / 3 GC, no ERCC.
# ------------------------------------------------------------
.glds_sfgc_dds <- function(counts_fname, metadata_fname) {
  data    <- file_imports(counts_fname, metadata_fname)
  counts  <- data$counts
  coldata <- data$coldata
  
  ercc <- grep("^ERCC", row.names(counts))
  if (length(ercc) > 0) counts <- counts[-ercc, ]
  counts <- parse_features(counts)
  
  coldata$spaceflight <- coldata$Factor.Value.Spaceflight. %>%
    { gsub(" ", "_", .) } %>% { gsub("#", "", .) }
  coldata <- coldata[!coldata$spaceflight %in%
                       c("Basal_Control", "Vivarium_Control", "Vivarium_control"), ]
  coldata$Sample.Name <- coldata$Sample.Name %>% { gsub("-", ".", .) } %>% factor()
  
  counts  <- counts[, colnames(counts) %in% coldata$Sample.Name]
  coldata <- coldata[match(colnames(counts), coldata$Sample.Name), ]
  coldata$spaceflight <- factor(coldata$spaceflight)
  row.names(coldata)  <- coldata$Sample.Name
  
  list(counts = counts, coldata = coldata)
}

glds270_dds <- function(counts_fname, metadata_fname) {
  built <- .glds_sfgc_dds(counts_fname, metadata_fname)
  dds <- DESeqDataSetFromMatrix(countData = round(built$counts), colData = built$coldata,
                                design = ~ spaceflight)
  list(data = dds, metadata = built$coldata)
}

glds599_dds <- function(counts_fname, metadata_fname) {
  built <- .glds_sfgc_dds(counts_fname, metadata_fname)
  dds <- DESeqDataSetFromMatrix(countData = round(built$counts), colData = built$coldata,
                                design = ~ spaceflight)
  list(data = dds, metadata = built$coldata)
}

glds580_dds <- function(counts_fname, metadata_fname) {
  built   <- .glds_sfgc_dds(counts_fname, metadata_fname)
  coldata <- built$coldata
  coldata$age        <- factor(ifelse(grepl("YNG", coldata$Sample.Name), "Young", "Old"))
  coldata$dissection <- factor(ifelse(grepl("ISS", coldata$Sample.Name), "ISS_T", "LAR"))
  
  design_formula <- if (BLOCK_580) ~ spaceflight + age + dissection else ~ spaceflight
  dds <- DESeqDataSetFromMatrix(countData = round(built$counts), colData = coldata,
                                design = design_formula)
  list(data = dds, metadata = coldata)
}

# ------------------------------------------------------------
# RUN DGE: subset to canonical SF/GC, fit, extract SF-vs-GC LFC
# ------------------------------------------------------------
run_dge <- function(study_output, out_csv, gc_levels = GC_LEVELS) {
  dds <- study_output$data
  

  keep <- as.character(dds$spaceflight) %in% c(SF_LEVEL, gc_levels)
  if (sum(keep) < ncol(dds))
    message(sprintf("  dropped %d sample(s) not in SF/GC sets", ncol(dds) - sum(keep)))
  dds <- dds[, keep]

  pooled <- ifelse(as.character(dds$spaceflight) == SF_LEVEL,
                   "Space_Flight", "Ground_Control")
  dds$spaceflight <- factor(pooled, levels = c("Ground_Control", "Space_Flight"))
  
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
  cat(sprintf("  %-28s SF=%d GC=%d  genes=%d  -> %s\n",
              out_csv,
              sum(dds$spaceflight == "Space_Flight"),
              sum(dds$spaceflight == "Ground_Control"),
              nrow(out), file.path(OUT_DIR, out_csv)))
}



run_dge(glds270_dds("GLDS-270_rna_seq_RSEM_Unnormalized_Counts", "270"), "glds270_fc_and_pvals.csv")
run_dge(glds580_dds("GLDS-573_rna_seq_RSEM_Unnormalized_Counts", "580"), "glds580_fc_and_pvals.csv")
run_dge(glds599_dds("GLDS-596_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq", "599"), "glds599_fc_and_pvals.csv")
