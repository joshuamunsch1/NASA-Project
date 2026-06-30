# ============================================================
#
#   inputs  (all already on disk):
#     17_integration_diagnostics.R                          (sourced for functions)
#     data/expression_results/glds*_fc_and_pvals.csv        (per-study DE -> referee)
#     data/model_free_enrichment/<TISSUE>_gsea_relaxed.tsv  (18_, the pooled cell)
#     data/go_bp_comparison_results/<id>_gseGO_high_pval.tsv (single studies)
#     data/go_bp_comparison_results/<arm>_<method>_gseGO_high_pval.tsv (panel)
#   outputs (alongside 17_'s, so the pooled row rbinds onto d2_discovery_summary):
#     data/go_bp_comparison_results/integration_diagnostics/
#         <TISSUE>_pooled_vs_referee_summary.csv        (the leaderboard)
#         <TISSUE>_pooled_merged_only_verdict.csv        (per-term verdicts)
# ============================================================

# ---- config: SET THESE TWO PER TISSUE ----------------------
TISSUE <- "heart"                                         # "liver" or "heart"
ROOT   <- if (TISSUE == "liver") {
  "path/to/liver"
} else {
  "path/to/heart"
}
setwd(ROOT)
DIAG_SCRIPT <- "17_integration_diagnostics.R"

INCLUDE_PANEL    <- TRUE     # also score every GSEA panel cell -> one leaderboard
SCORE_POOLED_ORA <- FALSE    # also score pooled-DE ORA vs the (GSEA) referee.
# exploratory only: namespace-matched (GO:BP) but NOT
# paradigm-matched, so it is a weaker check. OFF.
WRITE_PANEL_VERDICTS <- FALSE # if TRUE, per-term verdict file includes panel cells

# ---- per-tissue study sets (referee substrate + single-study union) ---------
# LIVER mirrors 17_'s STUDIES EXACTLY (6 studies) so the pooled cell is scored by
# the identical referee/union the classifier arms were. HEART uses the 3 cardiac
# per-study tables (10_dge_singlestudy_second_merged_dataset.R). `id` keys the
# single-study relaxed file (<id>_gseGO_high_pval.tsv); `rr` is a label; `expr`
# is the per-study DESeq2 table under data/expression_results/.
STUDIES_BY_TISSUE <- list(
  liver = list(
    list(rr = "RR1_NASA", id = "168_RR1", expr = "glds168rr1_fc_and_pvals.csv"),
    list(rr = "RR6",      id = "245",     expr = "glds245_fc_and_pvals.csv"),
    list(rr = "RR8",      id = "379",     expr = "glds379_fc_and_pvals.csv"),
    list(rr = "RR9",      id = "242",     expr = "glds242_fc_and_pvals.csv"),
    list(rr = "RR3",      id = "168_RR3", expr = "glds168rr3_fc_and_pvals.csv"),
    list(rr = "RR47",     id = "47",      expr = "glds47_fc_and_pvals.csv")
  ),
  heart = list(
    list(rr = "270", id = "270", expr = "glds270_fc_and_pvals.csv"),
    list(rr = "580", id = "580", expr = "glds580_fc_and_pvals.csv"),
    list(rr = "599", id = "599", expr = "glds599_fc_and_pvals.csv")
  )
)

# ---- source 17_ for the referee machinery (test mode = defs only) -----------
DIAG_TEST_MODE <- TRUE
if (!file.exists(DIAG_SCRIPT))
  stop("set DIAG_SCRIPT to your 17_integration_diagnostics.R (looked for '", DIAG_SCRIPT, "')")
source(DIAG_SCRIPT, local = FALSE)            # defines meta_analyze, gsego_relaxed,
# ensembl_to_entrez_ranking, read_study_de,
# split_core, load_relaxed, sig_ids,
# single_relaxed_path, merged_relaxed_path, ...
for (fn in c("meta_analyze", "gsego_relaxed", "ensembl_to_entrez_ranking",
             "read_study_de", "split_core", "load_relaxed", "sig_ids",
             "single_relaxed_path", "merged_relaxed_path"))
  if (!exists(fn, mode = "function"))
    stop("17_ did not provide ", fn, "() -- is DIAG_SCRIPT the right file?")

suppressPackageStartupMessages({
  library(clusterProfiler); library(org.Mm.eg.db); library(AnnotationDbi)
})


EXPR_DIR      <- "data/single_dge"
CMP_DIR       <- "data/go_bp_terms"
MF_DIR        <- "data/model_free_enrichment"
OUT_DIR       <- file.path(CMP_DIR, "integration_diagnostics")
SIG_CUT       <- 0.1            # p.adjust cut shared with 11/11b/13/17/18
D2_RANK_METRIC<- "meta_z"       # referee ranking (== 17_)
ORA_NAMESPACE <- "GO:BP"        # only bites tables with a 'source' col (ORA)
SEED          <- 12345
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

STUDIES <- STUDIES_BY_TISSUE[[TISSUE]]
if (is.null(STUDIES)) stop("no STUDIES defined for TISSUE='", TISSUE, "'")
# GSEA panel cells == the dense-model rows scorable by a meta-GSEA referee.
ARMS  <- c("zscore", "deseq2", "ruvg", "combat_ref")
PANEL_GSEA_METHODS <- c("svm", "glm")     

cat(sprintf("[%s] root=%s | %d studies for referee | SIG_CUT=%.2f\n",
            TISSUE, ROOT, length(STUDIES), SIG_CUT))

# ============================================================
# 1. Build the referee 
# ============================================================
de_list <- setNames(lapply(STUDIES, function(s) read_study_de(s$expr)),
                    vapply(STUDIES, function(s) s$rr, character(1)))
de_list <- Filter(Negate(is.null), de_list)
if (length(de_list) < 2)
  stop("need >=2 per-study DE tables under ", EXPR_DIR, " (found ", length(de_list), ")")
meta <- meta_analyze(de_list)
cat(sprintf("  meta-analysis over %d studies, %d genes (median I2 = %.2f)\n",
            length(de_list), nrow(meta), median(meta$I2, na.rm = TRUE)))
if (length(de_list) <= 3)


ens2ent <- AnnotationDbi::mapIds(org.Mm.eg.db, keys = meta$gene, keytype = "ENSEMBL",
                                 column = "ENTREZID", multiVals = "first")
het <- data.frame(ENTREZ = ens2ent, I2 = meta$I2, sign_agreement = meta$sign_agreement,
                  stringsAsFactors = FALSE)
het <- het[!is.na(het$ENTREZ) & !duplicated(het$ENTREZ), ]
rownames(het) <- het$ENTREZ

referee_sig <- function(meta_tab) {       
  rk  <- ensembl_to_entrez_ranking(meta_tab$gene, meta_tab[[D2_RANK_METRIC]])
  res <- gsego_relaxed(rk)
  list(relaxed = res, sig = res$ID[!is.na(res$p.adjust) & res$p.adjust <= SIG_CUT])
}
ref_full <- referee_sig(meta)
write.table(ref_full$relaxed,
            file.path(OUT_DIR, sprintf("%s_meta_gsea_referee.tsv", TISSUE)),
            sep = "\t", quote = TRUE, row.names = FALSE)
loso   <- lapply(seq_along(de_list), function(k) referee_sig(meta_analyze(de_list[-k]))$sig)
n_loso <- length(loso)
cat(sprintf("  meta-GSEA referee: %d significant terms (%d LOSO folds)\n",
            length(ref_full$sig), n_loso))

single_gsea_union <- unique(unlist(lapply(STUDIES, function(s)
  sig_ids(load_relaxed(single_relaxed_path(s$id, "gsea"), "gsea")))))
cat(sprintf("  single-study GSEA union: %d terms (across %d studies)\n",
            length(single_gsea_union), length(STUDIES)))

# ============================================================
# 2. Score any cell against the referee 
# ============================================================
score_cell <- function(approach, arm, method, merged) {
  msig        <- sig_ids(merged)
  merged_only <- setdiff(msig, single_gsea_union)
  vr <- list()
  for (id in merged_only) {
    row  <- merged[merged$ID == id, , drop = FALSE][1, ]
    core <- split_core(if ("core_enrichment" %in% colnames(row)) row$core_enrichment else NA)
    h    <- het[intersect(core, rownames(het)), , drop = FALSE]
    recovered <- id %in% ref_full$sig
    vr[[length(vr) + 1]] <- data.frame(
      Approach = approach, Arm = arm, Method = method, ID = id,
      Description = if ("Description" %in% colnames(row)) row$Description else NA,
      merged_NES = if ("NES" %in% colnames(row)) round(row$NES, 3) else NA_real_,
      meta_recovered = recovered,
      loso_survival  = if (recovered) sum(vapply(loso, function(s) id %in% s, logical(1))) else 0L,
      n_loso = n_loso,
      n_leadingedge_in_meta = nrow(h),
      median_I2_leadingedge = if (nrow(h)) round(median(h$I2), 3) else NA_real_,
      median_signagree_leadingedge = if (nrow(h)) round(median(h$sign_agreement), 3) else NA_real_,
      row.names = NULL, stringsAsFactors = FALSE)
  }
  verdict <- if (length(vr)) do.call(rbind, vr) else NULL
  if (!is.null(verdict))                                   # == 17_'s verdict rule
    verdict$verdict <- ifelse(!verdict$meta_recovered, "artifact_candidate",
                              ifelse(verdict$loso_survival >= verdict$n_loso - 1 &
                                       (is.na(verdict$median_I2_leadingedge) | verdict$median_I2_leadingedge <= 0.5),
                                     "credible_discovery", "fragile"))
  nrec <- sum(merged_only %in% ref_full$sig)
  summ <- data.frame(
    Approach = approach, Arm = arm, Method = method,
    n_sig = length(msig), n_merged_only = length(merged_only),
    n_meta_recovered = nrec,
    frac_recovered = if (length(merged_only)) round(nrec / length(merged_only), 3) else NA_real_,

    frac_all_sig_referee_backed = if (length(msig))
      round(sum(msig %in% ref_full$sig) / length(msig), 3) else NA_real_,
    row.names = NULL, stringsAsFactors = FALSE)
  list(summary = summ, verdict = verdict)
}

# ============================================================
# 3. pooled-DE + GSEA panel
# ============================================================
cells <- list()

mf_gsea <- load_relaxed(file.path(MF_DIR, sprintf("%s_gsea_relaxed.tsv", TISSUE)), "gsea")
if (is.null(mf_gsea))
  stop("pooled-DE GSEA relaxed not found at ",
       file.path(MF_DIR, sprintf("%s_gsea_relaxed.tsv", TISSUE)), " -- run 18_ first")
cells[["model_free_de"]] <- list(approach = "model_free_de", arm = "pooled",
                                 method = "model_free_de", merged = mf_gsea)

if (SCORE_POOLED_ORA) {
  mf_ora <- load_relaxed(file.path(MF_DIR, sprintf("%s_ora_relaxed.tsv", TISSUE)), "gsea")
  # NB: read as "gsea" on purpose -> no source-column filter; enrichGO is GO:BP only.
  if (!is.null(mf_ora))
    cells[["model_free_de_ORA"]] <- list(approach = "model_free_de_ORA", arm = "pooled",
                                         method = "model_free_de_ora", merged = mf_ora)
}

if (INCLUDE_PANEL) for (arm in ARMS) for (meth in PANEL_GSEA_METHODS) {
  d <- load_relaxed(merged_relaxed_path(arm, meth, "gsea"), "gsea"); if (is.null(d)) next
  key <- sprintf("%s__%s", arm, meth)
  cells[[key]] <- list(approach = key, arm = arm, method = meth, merged = d)
}
cat(sprintf("  scoring %d cells against the referee (pooled-DE + %d panel)\n",
            length(cells), length(cells) - 1L))

# ============================================================
# 4. run, rank and write data
# ============================================================
scored      <- lapply(cells, function(c) score_cell(c$approach, c$arm, c$method, c$merged))
summary_tab <- do.call(rbind, lapply(scored, `[[`, "summary"))
summary_tab <- summary_tab[order(-summary_tab$frac_recovered,
                                 -summary_tab$n_meta_recovered), ]
rownames(summary_tab) <- NULL
write.csv(summary_tab,
          file.path(OUT_DIR, sprintf("%s_pooled_vs_referee_summary.csv", TISSUE)),
          row.names = FALSE)

verdict_keys <- if (WRITE_PANEL_VERDICTS) names(scored) else
  intersect(c("model_free_de", "model_free_de_ORA"), names(scored))
verdict_tab  <- do.call(rbind, Filter(Negate(is.null),
                                      lapply(scored[verdict_keys], `[[`, "verdict")))
if (!is.null(verdict_tab)) {
  rownames(verdict_tab) <- NULL
  write.csv(verdict_tab,
            file.path(OUT_DIR, sprintf("%s_pooled_merged_only_verdict.csv", TISSUE)),
            row.names = FALSE)
}

