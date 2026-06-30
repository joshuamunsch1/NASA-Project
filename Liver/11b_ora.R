# ============================================================
# 11b_ora.R  
# ============================================================

suppressPackageStartupMessages({
  library(gprofiler2)
})

setwd("path/to/liver")

# ---- config -------------------------------------------------
SCORES_DIR      <- "."
EXPR_DIR        <- "data/single_dge"
BACKGROUND_FILE <- "prefiltered_pseudogenes.csv"   
ORA_TOPK        <- 500                         
ORGANISM        <- "mmusculus"
CORRECTION      <- "g_SCS"
DOMAIN          <- "custom"

for (d in c("data/go_bp_terms",
            "data/go_terms_merged",
            "data/go_terms_single_datasets"))
  dir.create(d, showWarnings = FALSE, recursive = TRUE)

ARMS <- list(
  list(arm = "zscore",     tag = "zscore"),
  list(arm = "deseq2",     tag = "deseq2"),
  list(arm = "ruvg",       tag = "ruvg"),
  list(arm = "combat_ref", tag = "combat")
)
ORA_MODELS <- list(
  list(name = "elasticnet", source = "importance"),
  list(name = "xgboost",    source = "importance"),
  list(name = "glm",         source = "importance"),
  list(name = "svm",         source = "importance")
)

# ---- background --------------------------------------------
read_background <- function() {
  if (!file.exists(BACKGROUND_FILE)) stop("Background file not found: ", BACKGROUND_FILE)
  feats <- read.table(BACKGROUND_FILE, header = TRUE, sep = ",", stringsAsFactors = FALSE)
  if (ncol(feats) < 2) stop("Expected >=2 columns in ", BACKGROUND_FILE, " (gene ids in column 2)")
  bg <- unique(as.character(feats[, 2]))           # ENSEMBL ids (== counts rownames)
  bg[!is.na(bg) & nzchar(bg)]
}
BACKGROUND <- read_background()

empty_cols <- function()
  data.frame(ID = character(), Description = character(), p.adjust = numeric(),
             intersection_size = integer(), term_size = integer(),
             source = character(), stringsAsFactors = FALSE)

# ---- ORA via g:Profiler -------------------------------
run_ora <- function(query_genes, relaxed_path, sig_path, label = "") {
  query_genes <- intersect(unique(as.character(query_genes)), BACKGROUND)
  if (length(query_genes) < 1) {
    write.table(empty_cols(), relaxed_path, sep = "\t", quote = TRUE, row.names = FALSE)
    write.table(empty_cols(), sig_path,     sep = "\t", quote = TRUE, row.names = FALSE)
  }
  res_obj <- tryCatch(
    gost(query              = query_genes,
         organism           = ORGANISM,
         ordered_query      = TRUE,
         significant        = TRUE,          # return ALL tested terms (relaxed scope)
         user_threshold     = 0.05,
         correction_method  = CORRECTION,
         domain_scope       = DOMAIN,
         custom_bg          = BACKGROUND,
         
         evcodes            = FALSE),
    error = function(e) { cat(sprintf("  %-26s gost failed: %s\n", label, conditionMessage(e))); NULL })
  
  res <- if (is.null(res_obj)) NULL else res_obj$result
  if (is.null(res) || !nrow(res)) {
    write.table(empty_cols(), relaxed_path, sep = "\t", quote = TRUE, row.names = FALSE)
    write.table(empty_cols(), sig_path,     sep = "\t", quote = TRUE, row.names = FALSE)
  }
  
  out <- data.frame(
    ID                = as.character(res$term_id),
    Description       = as.character(res$term_name),
    p.adjust          = as.numeric(res$p_value),
    intersection_size = as.integer(res$intersection_size),
    term_size         = as.integer(res$term_size),
    source            = as.character(res$source),
    stringsAsFactors  = FALSE)
  write.table(out, relaxed_path, sep = "\t", quote = TRUE, row.names = FALSE)
  sig <- out[!is.na(out$p.adjust) & out$p.adjust <= 0.1, ]
  write.table(sig, sig_path, sep = "\t", quote = TRUE, row.names = FALSE)
  invisible(out)
}

# ---- selection helpers -------------------------------------
imp_path <- function(model, tag) {
  suffix <- if (nzchar(tag)) paste0("_", tag) else ""
  file.path(SCORES_DIR, sprintf("%s_shap_importance%s.csv", model, suffix))
}
mrmr_path <- function(tag) {
  suffix <- if (nzchar(tag)) paste0("_", tag) else ""
  cand <- c(file.path(SCORES_DIR, sprintf("mrmr%s.csv", suffix)),
            file.path(SCORES_DIR, "mrmr.csv"))
  hit <- cand[file.exists(cand)]
  if (length(hit)) hit[1] else NA_character_
}
top_k_genes <- function(path, k) {
  tab <- read.csv(path, header = TRUE, row.names = 1, stringsAsFactors = FALSE)
  col <- if ("shap_mean" %in% colnames(tab)) "shap_mean" else colnames(tab)[1]
  ord <- order(abs(tab[[col]]), decreasing = TRUE)
  # tab_d = tab[order(abs(tab$shap_mean), decreasing = TRUE),]
  rownames(tab)[ord[seq_len(min(k, nrow(tab)))]]
}

# ------------------------------------------------------------
# MERGED -> arm x ORA-model
# ------------------------------------------------------------
cat(sprintf("ORA arm: merged-study g:Profiler (top-%d genes, %s background)\n",
            ORA_TOPK, BACKGROUND_FILE))
for (a in ARMS) for (mdl in ORA_MODELS) {
  if (mdl$source == "importance") {
    p <- imp_path(mdl$name, a$tag)
    if (!file.exists(p)) { cat("  SKIP (missing)", basename(p), "\n"); next }
    genes <- top_k_genes(p, ORA_TOPK)
  } else {  # genelist (mRMR)
    gp <- mrmr_path(a$tag)
    if (is.na(gp)) { cat(sprintf("  SKIP mrmr (no list) for %s\n", a$arm)); next }
    gl <- read.csv(gp, header = TRUE, stringsAsFactors = FALSE)
    gcol <- if ("Gene" %in% colnames(gl)) "Gene" else colnames(gl)[ncol(gl)]
    genes <- as.character(gl[[gcol]])
  }
  run_ora(genes,
          file.path("data/go_bp_comparison_results", sprintf("%s_%s_ora_high_pval.tsv", a$arm, mdl$name)),
          file.path("data/combined_study_go_terms",  sprintf("%s_%s_ora.tsv",          a$arm, mdl$name)),
          label = sprintf("%s/%s", a$arm, mdl$name))
}

# ------------------------------------------------------------
# SINGLE-STUDY ORA
# ------------------------------------------------------------
ORA_RANK_METRIC <- "stat"
studies <- list(
  list(expr = "glds47_fc_and_pvals.csv",     id = "47",      sig = "GLDS-47-ORA.tsv"),
  list(expr = "glds168rr1_fc_and_pvals.csv", id = "168_RR1", sig = "GLDS-168_RR1-ORA.tsv"),
  list(expr = "glds168rr3_fc_and_pvals.csv", id = "168_RR3", sig = "GLDS-168_RR3-ORA.tsv"),
  list(expr = "glds242_fc_and_pvals.csv",    id = "242",     sig = "GLDS-242-ORA.tsv"),
  list(expr = "glds245_fc_and_pvals.csv",    id = "245",     sig = "GLDS-245-ORA.tsv"),
  list(expr = "glds379_fc_and_pvals.csv",    id = "379",     sig = "GLDS-379-ORA.tsv")
)
for (s in studies) {
  expr_path <- file.path(EXPR_DIR, s$expr)
  if (!file.exists(expr_path)) { cat("  SKIP (missing)", s$expr, "\n"); next }
  fc  <- read.csv(expr_path, header = TRUE, stringsAsFactors = FALSE)
  rank_col <- if (ORA_RANK_METRIC == "stat" && "stat" %in% colnames(fc)) "stat" else "log2fc"
  sel <- fc[!is.na(fc[[rank_col]]), ]
  ord <- order(abs(sel[[rank_col]]), decreasing = TRUE)
  top_ens <- sel$gene[ord[seq_len(min(ORA_TOPK, nrow(sel)))]]
  run_ora(top_ens,
          file.path("data/go_bp_comparison_results", sprintf("%s_ora_high_pval.tsv", s$id)),
          file.path("data/single_study_go_terms",    s$sig),
          label = sub("_fc.*", "", s$expr))
}

