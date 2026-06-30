# ============================================================
# 14_power_permutation_loso.R
#
#   inputs : pipeline/merged_counts.rds   (genes x samples, raw)
#            pipeline/meta.rds            ($study, $condition)
#   outputs: data/framework_tests/reference_signature.csv   (the full-data
#                                          reference term set scored against)
#            data/framework_tests/loso.csv
#            data/framework_tests/permutation_null.csv
#            data/framework_tests/permutation_loso.png       (if ggplot2 available)
# ============================================================

suppressPackageStartupMessages({
  library(edgeR); library(glmnet); library(clusterProfiler)
  library(org.Mm.eg.db); library(dplyr)
})

setwd("path/to/liver")

# ---- config -------------------------------------------------
MODEL            <- "glm"                 # "glm" (ridge LR) or "svm" (linear SVM)
PADJ_CUT         <- 0.1                    # significance threshold (matches 11_gsea.R)
B_PERM           <- 200
SEED             <- 12345
OUT_DIR          <- "data/framework_tests"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
set.seed(SEED)

# ---- shared route -------------------------------------------
zscore_by_study <- function(logmat, study) {
  out <- logmat
  for (s in unique(study)) {
    cols <- which(study == s)
    sub  <- logmat[, cols, drop = FALSE]
    mu   <- rowMeans(sub)
    sdv  <- apply(sub, 1, sd); sdv[!is.finite(sdv) | sdv == 0] <- 1
    out[, cols] <- (sub - mu) / sdv
  }
  out
}

ens_to_entrez_ranking <- function(ens_ids, scores) {
  df <- data.frame(ENSEMBL = ens_ids, score = scores, stringsAsFactors = FALSE)
  df$ENTREZID <- mapIds(org.Mm.eg.db, keys = df$ENSEMBL, keytype = "ENSEMBL",
                        column = "ENTREZID", multiVals = "first")
  df <- df[!is.na(df$ENTREZID) & !duplicated(df$ENTREZID), ]
  v <- df$score; names(v) <- df$ENTREZID
  sort(v, decreasing = TRUE)
}

signed_ranking <- function(corrected, y) {
  x <- t(corrected)                                  # samples x genes
  if (MODEL == "glm") {
    fit  <- cv.glmnet(x, y, family = "binomial", alpha = 0, nfolds = 5)
    cf   <- as.numeric(coef(fit, s = "lambda.1se"))[-1]
    names(cf) <- colnames(x)
  } else {                                           # linear SVM weights
    if (!requireNamespace("e1071", quietly = TRUE)) stop("install e1071 for MODEL='svm'")
    m  <- e1071::svm(x, y, kernel = "linear", scale = FALSE, probability = FALSE)
    w  <- t(m$coefs) %*% m$SV
    cf <- as.numeric(w); names(cf) <- colnames(x)
  }
  cf
}

run_route <- function(counts_sub, meta_sub) {
  keep   <- rowSums(counts_sub) > 0
  counts_sub <- counts_sub[keep, , drop = FALSE]
  dge    <- calcNormFactors(DGEList(counts = counts_sub), method = "TMM")
  logcpm <- cpm(dge, log = TRUE, prior.count = 1)
  corr   <- zscore_by_study(logcpm, as.character(meta_sub$study))
  v      <- apply(corr, 1, var); corr <- corr[is.finite(v) & v > 0, , drop = FALSE]
  y      <- factor(meta_sub$condition)
  cf     <- signed_ranking(corr, y)
  rank   <- ens_to_entrez_ranking(names(cf), cf)
  gse <- gseGO(geneList = rank, ont = "BP", keyType = "ENTREZID", exponent = 1,
               eps = 0, pvalueCutoff = 0.9, pAdjustMethod = "BH",
               OrgDb = org.Mm.eg.db, seed = TRUE, verbose = FALSE)
  res <- gse@result
  res$ID[res$p.adjust <= PADJ_CUT]
}

# ---- load -----------------------------------------------------
counts <- readRDS("pipeline/merged_counts.rds")
meta   <- readRDS("pipeline/meta.rds")
stopifnot(all(c("study", "condition") %in% colnames(meta)))
meta$study <- as.character(meta$study)
N <- ncol(counts)
cat(sprintf("Merged matrix: %d genes x %d samples, %d studies\n",
            nrow(counts), N, length(unique(meta$study))))

# ----  the route full-data signature ---------
reference_ids <- run_route(counts, meta)
stopifnot(length(reference_ids) > 0)
cat(sprintf("  reference: %d significant GO:BP terms (p.adj <= %.2f)\n",
            length(reference_ids), PADJ_CUT))
write.csv(data.frame(ID = reference_ids),
          file.path(OUT_DIR, "reference_signature.csv"), row.names = FALSE)
recovery <- function(ids) 100 * length(intersect(ids, reference_ids)) / length(reference_ids)

# ---- #1 LOSO ------------------------------------------------
loso <- do.call(rbind, lapply(unique(meta$study), function(s) {
  idx <- which(meta$study != s)
  ids <- tryCatch(run_route(counts[, idx, drop = FALSE], meta[idx, ]), error = function(e) NULL)
  if (is.null(ids)) return(NULL)
  cat(sprintf("  drop %-10s n=%d  n_sig=%d  recovery=%.1f%%\n",
              s, length(idx), length(ids), recovery(ids)))
  data.frame(left_out = s, n = length(idx), n_sig = length(ids), recovery = recovery(ids))
}))
write.csv(loso, file.path(OUT_DIR, "loso.csv"), row.names = FALSE)

# ---- #2 PERMUTATION NULL ------------------------------------
permute_within_study <- function(study, cond) {
  out <- cond
  for (s in unique(study)) { m <- which(study == s); out[m] <- sample(cond[m]) }
  out
}

observed_ids <- reference_ids
obs_nsig <- length(observed_ids); obs_rec <- recovery(observed_ids)
perm <- list()
for (b in seq_len(B_PERM)) {
  mp <- meta; mp$condition <- permute_within_study(meta$study, as.character(meta$condition))
  ids <- tryCatch(run_route(counts, mp), error = function(e) NULL)
  if (is.null(ids)) next
  perm[[length(perm) + 1]] <- data.frame(replicate = b, n_sig = length(ids),
                                         recovery = recovery(ids))
  if (b %% 20 == 0) cat(sprintf("  perm %d/%d\n", b, B_PERM))
}
perm_df <- do.call(rbind, perm)
emp_p_nsig <- (1 + sum(perm_df$n_sig    >= obs_nsig)) / (1 + nrow(perm_df))
emp_p_rec  <- (1 + sum(perm_df$recovery >= obs_rec )) / (1 + nrow(perm_df))
perm_df <- rbind(data.frame(replicate = 0, n_sig = obs_nsig, recovery = obs_rec), perm_df)
attr(perm_df, "obs") <- c(n_sig = obs_nsig, recovery = obs_rec,
                          emp_p_nsig = emp_p_nsig, emp_p_recovery = emp_p_rec)
write.csv(perm_df, file.path(OUT_DIR, "permutation_null.csv"), row.names = FALSE)


