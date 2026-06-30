# ============================================================
# 11_gsea.R 
# ============================================================

suppressPackageStartupMessages({
  library(clusterProfiler)
  organism <- "org.Mm.eg.db"
  library(organism, character.only = TRUE)
  library(dplyr)
})

setwd("path/to/liver")

# ---- config -------------------------------------------------
SCORES_DIR <- "feature_importance"                     
EXPR_DIR   <- "data/single_dge"
SEED       <- 12345

for (d in c("data/go_bp_terms",
            "data/go_terms_merged",
            "data/go_terms_single_datasets"))
  dir.create(d, showWarnings = FALSE, recursive = TRUE)

# GSEA arm: dense, signed models only.
ARMS <- list(
  list(arm = "zscore",     tag = "zscore"),
  list(arm = "deseq2",     tag = "deseq2"),
  list(arm = "ruvg",       tag = "ruvg"),
  list(arm = "combat_ref", tag = "combat")
)
GSEA_MODELS <- c('elasticnet', 'xgboost', 'svm', 'glm')

# ------------------------------------------------------------
run_gsea <- function(scores, high_pval_path, sig_path, label = "") {
  if (!is.null(SEED)) set.seed(SEED)
  gse <- gseGO(geneList      = scores,
               ont           = "BP",
               keyType       = "ENTREZID",
               exponent      = 1,
               eps           = 0,
               pvalueCutoff  = 0.9,
               pAdjustMethod = "BH",
               OrgDb         = org.Mm.eg.db,
               seed          = !is.null(SEED))
  write.table(gse, high_pval_path, sep = "\t", quote = TRUE, row.names = FALSE)
  sig <- dplyr::filter(gse@result, p.adjust <= 0.1)
  write.table(sig, sig_path, sep = "\t", quote = TRUE, row.names = FALSE)
  cat(sprintf("  %-22s relaxed(<=0.9)=%4d  significant(<=0.1)=%4d\n",
              label, nrow(gse@result), nrow(sig)))
  invisible(gse)
}

ensembl_to_entrez_ranking <- function(ens_ids, scores) {
  df <- data.frame(ENSEMBL = ens_ids, score = scores, stringsAsFactors = FALSE)
  df$ENTREZID <- mapIds(org.Mm.eg.db, keys = df$ENSEMBL,
                        keytype = "ENSEMBL", column = "ENTREZID",
                        multiVals = "first")
  df <- df[!is.na(df$ENTREZID), ]
  df <- df[!duplicated(df$ENTREZID), ]
  df <- df %>% arrange(desc(score))
  v <- df$score; names(v) <- df$ENTREZID
  sort(v, decreasing = TRUE)
}

score_file <- function(model, tag) {
  suffix <- if (nzchar(tag)) paste0("_", tag) else ""
  file.path(SCORES_DIR, sprintf("%s_importances%s.csv", model, suffix))
}

# ------------------------------------------------------------
# MERGED -> arm x model  
# ------------------------------------------------------------
cat("GSEA arm: merged-study gseGO (dense signed coefficients)\n")
for (a in ARMS) for (model in GSEA_MODELS) {
  p <- score_file(model, a$tag)
  if (!file.exists(p)) { cat("  SKIP (missing)", basename(p), "\n"); next }
  
  tab <- read.csv(p, header = TRUE, row.names = 1, stringsAsFactors = FALSE)
  col <- if ("Score" %in% colnames(tab)) "Score" else colnames(tab)[1]
  if (all(tab[[col]] >= 0, na.rm = TRUE))
    cat(sprintf("    [warn] %s/%s coefficients are non-negative; gseGO expects a signed ranking.\n",
                a$arm, model))
  rank <- ensembl_to_entrez_ranking(rownames(tab), tab[[col]])
  
  relaxed_out <- file.path("data/go_bp_comparison_results",
                           sprintf("%s_%s_gseGO_high_pval.tsv", a$arm, model))
  sig_out     <- file.path("data/combined_study_go_terms",
                           sprintf("%s_%s_gseGO.tsv", a$arm, model))
  run_gsea(rank, relaxed_out, sig_out, label = sprintf("%s/%s", a$arm, model))
  
  if (a$arm == "zscore" && model == "svm") {
    file.copy(relaxed_out, "data/go_bp_comparison_results/svm_gseGO_high_pval.tsv", overwrite = TRUE)
    file.copy(sig_out,     "data/combined_study_go_terms/svm_coeff_gseGO.tsv",      overwrite = TRUE)
  }
}

# ------------------------------------------------------------
# ranked DESeq2 log2FC single studies
# ------------------------------------------------------------
studies <- list(
  list(expr = "glds47_fc_and_pvals.csv",     high = "47_gseGO_high_pval.tsv",      sig = "GLDS-47-GO.tsv"),
  list(expr = "glds168rr1_fc_and_pvals.csv", high = "168_RR1_gseGO_high_pval.tsv", sig = "GLDS-168_RR1-GO.tsv"),
  list(expr = "glds168rr3_fc_and_pvals.csv", high = "168_RR3_gseGO_high_pval.tsv", sig = "GLDS-168_RR3-GO.tsv"),
  list(expr = "glds242_fc_and_pvals.csv",    high = "242_gseGO_high_pval.tsv",     sig = "GLDS-242-GO.tsv"),
  list(expr = "glds245_fc_and_pvals.csv",    high = "245_gseGO_high_pval.tsv",     sig = "GLDS-245-GO.tsv"),
  list(expr = "glds379_fc_and_pvals.csv",    high = "379_gseGO_high_pval.tsv",     sig = "GLDS-379-GO.tsv")
)

for (s in studies) {
  expr_path <- file.path(EXPR_DIR, s$expr)
  if (!file.exists(expr_path)) { cat("  SKIP (missing)", s$expr, "\n"); next }
  fc <- read.csv(expr_path, header = TRUE, stringsAsFactors = FALSE)
  rank <- ensembl_to_entrez_ranking(fc$gene, fc$log2fc)
  run_gsea(rank,
           file.path("data/go_bp_comparison_results", s$high),
           file.path("data/single_study_go_terms",   s$sig),
           label = sub("_fc.*", "", s$expr))
}

