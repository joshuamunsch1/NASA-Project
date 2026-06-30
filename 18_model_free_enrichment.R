# ============================================================

#   inputs : pipeline/merged_counts.rds   (genes x samples, raw)   [== 14_]
#            pipeline/meta.rds            ($study, $condition)     [== 14_]
#   outputs: data/model_free_enrichment/<TISSUE>_deseq2_de.csv
#            data/model_free_enrichment/<TISSUE>_gsea_relaxed.tsv  (p.adjust<=0.9)
#            data/model_free_enrichment/<TISSUE>_gsea_sig.tsv      (p.adjust<=0.1)
#            data/model_free_enrichment/<TISSUE>_ora_relaxed.tsv   (all-direction full table)
#            data/model_free_enrichment/<TISSUE>_ora_sig.tsv       (stacked all/up/down, p.adjust<=0.1)
#            data/model_free_enrichment/<TISSUE>_enrichment_summary.csv  (the row you compare)

# ============================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(clusterProfiler)
  library(org.Mm.eg.db)
  library(dplyr)
})

TISSUE       <- "heart"                                  # "liver" or "heart" 
PROJECT_ROOT   <- if (TISSUE == "liver") {
  "path/to/liver"
} else {
  "path/to/heart"
}
setwd(PROJECT_ROOT)

DESIGN_COVAR <- "study"   
RANK_METRIC  <- "stat"   
SIG_PADJ     <- 0.1       
ORA_INPUT    <- "padj"    
ORA_TOPK     <- 500       
MIN_GS       <- 10        
MAX_GS       <- 500
SEED         <- 12345
OUT_DIR      <- "data/model_free_enrichment"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- cross-approach comparison config (Section 6) -----------
# Scores the model-free pooled-DE result as ONE MORE "approach" against (a) the
# single studies and (b) the integration -> label-prediction -> coefficient/top-k
# -> GSEA/ORA panel, using the SAME relaxed files and p.adjust<=SIG_CUT rule as
# 13_go_overlap_methods.R, so the model-free cell is measured in identical units
# to every method cell. 
RUN_COMPARISON        <- TRUE
CMP_DIR               <- "data/go_bp_terms"  
SIG_CUT               <- SIG_PADJ                          
WRITE_MF_PAIR_DETAILS <- TRUE                              

RESTRICT_ORA_TO_GOBP  <- TRUE
ARMS  <- c("zscore", "deseq2", "ruvg", "combat_ref")
PANEL <- list(   # (method, paradigm) cells -- identical routing to 13_go_overlap_methods.R
  list(method = "svm",        paradigm = "gsea"), list(method = "svm",        paradigm = "ora"),
  list(method = "glm",        paradigm = "gsea"), list(method = "glm",        paradigm = "ora"),
  list(method = "elasticnet", paradigm = "gsea"), list(method = "elasticnet", paradigm = "ora"),
  list(method = "xgboost",    paradigm = "gsea"), list(method = "xgboost",    paradigm = "ora")
)


STUDIES <- if (TISSUE == "heart") {
  list(list(rr = "270", id = "270"), list(rr = "580", id = "580"), list(rr = "599", id = "599"))
} else {
  list(list(rr = "RR1_NASA", id = "168_RR1"), list(rr = "RR6", id = "245"),
       list(rr = "RR8", id = "379"),          list(rr = "RR9", id = "242"))
}

BACKGROUND_FILE      <- "prefiltered_pseudogenes.csv"   

# ------------------------------------------------------------
#  reused from 11_gsea.R 
# ------------------------------------------------------------
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

run_gsea <- function(scores, relaxed_path, sig_path, label = "") {
  if (!is.null(SEED)) set.seed(SEED)
  gse <- gseGO(geneList      = scores,
               ont           = "BP",
               keyType       = "ENTREZID",
               exponent      = 1,
               eps           = 0,
               minGSSize     = MIN_GS,
               maxGSSize     = MAX_GS,
               pvalueCutoff  = 0.9,
               pAdjustMethod = "BH",
               OrgDb         = org.Mm.eg.db,
               seed          = !is.null(SEED))
  res <- as.data.frame(gse@result)
  write.table(res, relaxed_path, sep = "\t", quote = TRUE, row.names = FALSE)
  sig <- res[res$p.adjust <= SIG_PADJ, , drop = FALSE]
  write.table(sig, sig_path, sep = "\t", quote = TRUE, row.names = FALSE)
  cat(sprintf("  GSEA %-5s relaxed(<=0.9)=%4d  significant(<=%.2f)=%4d\n",
              label, nrow(res), SIG_PADJ, nrow(sig)))
  sig
}

ens_to_entrez <- function(ens_ids) {
  e <- mapIds(org.Mm.eg.db, keys = unique(as.character(ens_ids)),
              keytype = "ENSEMBL", column = "ENTREZID", multiVals = "first")
  unique(e[!is.na(e)])
}

run_ora <- function(query_entrez, universe_entrez, relaxed_path, sig_path, label = "") {
  query_entrez <- intersect(query_entrez, universe_entrez)
  if (length(query_entrez) < 1) {
    cat(sprintf("  ORA  %-5s no query genes -> 0 terms\n", label))
    return(data.frame(ID = character(), Description = character(),
                      p.adjust = numeric(), stringsAsFactors = FALSE))
  }
  eg <- enrichGO(gene          = query_entrez,
                 universe      = universe_entrez,
                 OrgDb         = org.Mm.eg.db,
                 keyType       = "ENTREZID",
                 ont           = "BP",
                 minGSSize     = MIN_GS,
                 maxGSSize     = MAX_GS,
                 pvalueCutoff  = 1,    # keep full table; threshold on p.adjust below
                 qvalueCutoff  = 1,    # disable separate q filter -> single BH rule (== GSEA)
                 pAdjustMethod = "BH",
                 readable      = FALSE)
  res <- if (is.null(eg)) data.frame() else as.data.frame(eg@result)
  if (!nrow(res)) {
    cat(sprintf("  ORA  %-5s tested=0  (query=%d)\n", label, length(query_entrez)))
    return(res)
  }
  res$gene_set <- label
  if (nzchar(relaxed_path)) write.table(res, relaxed_path, sep = "\t", quote = TRUE, row.names = FALSE)
  sig <- res[res$p.adjust <= SIG_PADJ, , drop = FALSE]
  if (nzchar(sig_path)) write.table(sig, sig_path, sep = "\t", quote = TRUE, row.names = FALSE)
  cat(sprintf("  ORA  %-5s tested=%4d  significant(<=%.2f)=%4d  (query=%d)\n",
              label, nrow(res), SIG_PADJ, nrow(sig), length(query_entrez)))
  sig
}

gini <- function(x) {
  x <- sort(abs(x[is.finite(x)]))
  n <- length(x); if (n < 2L || sum(x) == 0) return(NA_real_)
  sum((2 * seq_len(n) - n - 1) * x) / (n * sum(x))
}

# ============================================================
# 1. POOLED DESeq2 
# ============================================================
counts <- readRDS("pipeline/merged_counts.rds")
meta   <- readRDS("pipeline/meta.rds")
stopifnot(all(c("study", "condition") %in% colnames(meta)))
stopifnot(all(colnames(counts) %in% rownames(meta)))
meta <- meta[colnames(counts), , drop = FALSE]           

meta$study <- factor(as.character(meta$study))
stopifnot(all(c("GC", "SF") %in% levels(factor(as.character(meta$condition)))))  # expects SF/GC encoding
meta$condition <- relevel(factor(as.character(meta$condition)), ref = "GC")      # +ve = up in flight

counts <- round(as.matrix(counts))
counts <- counts[rowSums(counts) > 0, , drop = FALSE]

design <- if (nzchar(DESIGN_COVAR) && nlevels(meta[[DESIGN_COVAR]]) > 1)
  as.formula(sprintf("~ %s + condition", DESIGN_COVAR)) else ~ condition
cat(sprintf("[%s] DESeq2 %s | %d genes x %d samples | %d studies\n",
            TISSUE, deparse(design), nrow(counts), ncol(counts), nlevels(meta$study)))

dds <- DESeqDataSetFromMatrix(countData = counts, colData = meta, design = design)
dds <- DESeq(dds)
res <- results(dds, contrast = c("condition", "SF", "GC"))   # log2(SF/GC); +ve = up in flight

de <- data.frame(ENSEMBL = rownames(res),
                 log2fc  = res$log2FoldChange,
                 lfcSE   = res$lfcSE,
                 stat    = res$stat,
                 pvalue  = res$pvalue,
                 padj    = res$padj,
                 stringsAsFactors = FALSE)
de$ENTREZID <- mapIds(org.Mm.eg.db, keys = de$ENSEMBL, keytype = "ENSEMBL",
                      column = "ENTREZID", multiVals = "first")
write.csv(de, file.path(OUT_DIR, sprintf("%s_deseq2_de.csv", TISSUE)), row.names = FALSE)

# ============================================================
# 2. GSEA (GO:BP)
# ============================================================
rank_vals <- if (RANK_METRIC == "stat") de$stat else de$log2fc
ok        <- is.finite(rank_vals)
gsea_rank <- ensembl_to_entrez_ranking(de$ENSEMBL[ok], rank_vals[ok])
cat(sprintf("[%s] GSEA ranking: %d genes (metric=%s)\n", TISSUE, length(gsea_rank), RANK_METRIC))
gsea_sig <- run_gsea(gsea_rank,
                     file.path(OUT_DIR, sprintf("%s_gsea_relaxed.tsv", TISSUE)),
                     file.path(OUT_DIR, sprintf("%s_gsea_sig.tsv",     TISSUE)),
                     label = TISSUE)

# ============================================================
# 3. ORA 
# ============================================================
universe <- ens_to_entrez(de$ENSEMBL[is.finite(de$padj)])  

select_genes <- function(direction) {
  d <- de[is.finite(de$padj), ]
  if (ORA_INPUT == "topk") {
    d <- d[is.finite(d$stat), ]
    d <- d[order(abs(d$stat), decreasing = TRUE), ]
    if (direction == "up")   d <- d[d$stat > 0, ]
    if (direction == "down") d <- d[d$stat < 0, ]
    sel <- head(d$ENSEMBL, ORA_TOPK)
  } else {                                  
    d <- d[d$padj <= SIG_PADJ, ]
    if (direction == "up")   d <- d[d$log2fc > 0, ]
    if (direction == "down") d <- d[d$log2fc < 0, ]
    sel <- d$ENSEMBL
  }
  ens_to_entrez(sel)
}

ora_all  <- run_ora(select_genes("all"),  universe,
                    file.path(OUT_DIR, sprintf("%s_ora_relaxed.tsv", TISSUE)),
                    file.path(OUT_DIR, sprintf("%s_ora_sig.tsv",     TISSUE)), label = "all")
ora_up   <- run_ora(select_genes("up"),   universe, "", "", label = "up")
ora_down <- run_ora(select_genes("down"), universe, "", "", label = "down")

ora_sig_all <- do.call(rbind, Filter(NROW, list(ora_all, ora_up, ora_down)))
if (NROW(ora_sig_all))
  write.table(ora_sig_all, file.path(OUT_DIR, sprintf("%s_ora_sig.tsv", TISSUE)),
              sep = "\t", quote = TRUE, row.names = FALSE)

# ============================================================
# 4. comparison summary
# ============================================================
gsea_ids <- if (NROW(gsea_sig)) gsea_sig$ID else character(0)
ora_ids  <- if (NROW(ora_all))  ora_all$ID  else character(0)   
inter    <- intersect(gsea_ids, ora_ids)
uni      <- union(gsea_ids, ora_ids)

n_de_up   <- sum(de$padj <= SIG_PADJ & de$log2fc > 0, na.rm = TRUE)
n_de_down <- sum(de$padj <= SIG_PADJ & de$log2fc < 0, na.rm = TRUE)

n_gsea_up   <- if (NROW(gsea_sig)) sum(gsea_sig$NES > 0, na.rm=TRUE) else 0L
n_gsea_down <- if (NROW(gsea_sig)) sum(gsea_sig$NES < 0, na.rm=TRUE) else 0L

summary_row <- data.frame(
  tissue           = TISSUE,
  n_samples        = ncol(counts),
  n_studies        = nlevels(meta$study),
  n_genes_tested   = sum(is.finite(de$padj)),
  rank_metric      = RANK_METRIC,
  gini_abs_rank    = round(gini(rank_vals), 3),    
  n_de_sig         = n_de_up + n_de_down,
  n_de_up          = n_de_up,
  n_de_down        = n_de_down,
  n_gsea_sig       = length(gsea_ids),
  n_gsea_up        = n_gsea_up,
  n_gsea_down      = n_gsea_down,
  n_ora_sig_all    = length(ora_ids),
  n_ora_sig_up     = if (NROW(ora_up))   nrow(ora_up)   else 0L,
  n_ora_sig_down   = if (NROW(ora_down)) nrow(ora_down) else 0L,
  gsea_ora_shared  = length(inter),
  gsea_ora_jaccard = if (length(uni)) round(length(inter) / length(uni), 3) else NA_real_,
  stringsAsFactors = FALSE)
write.csv(summary_row, file.path(OUT_DIR, sprintf("%s_enrichment_summary.csv", TISSUE)),
          row.names = FALSE)


# ============================================================
# 6. OVERLAP WITH SINGLE STUDIES + COMPARISON WITH THE OTHER APPROACHES
# ============================================================
if (RUN_COMPARISON) {

  rd <- function(p) read.table(p, sep = "\t", header = TRUE, quote = "\"", stringsAsFactors = FALSE)
  relx <- function(paradigm) if (paradigm == "gsea") "gseGO_high_pval" else "ora_high_pval"
  merged_relaxed_path <- function(arm, method, paradigm)
    file.path(CMP_DIR, sprintf("%s_%s_%s.tsv", arm, method, relx(paradigm)))
  single_relaxed_path <- function(id, paradigm)
    file.path(CMP_DIR, sprintf("%s_%s.tsv", id, relx(paradigm)))
  
  gobp_filter <- function(df) {
    if (RESTRICT_ORA_TO_GOBP && !is.null(df) && "source" %in% colnames(df))
      df[df$source == "GO:BP", , drop = FALSE] else df
  }
  load_relaxed <- function(path) {
    if (!file.exists(path)) return(NULL)
    df <- rd(path)
    if (!nrow(df) || !all(c("ID", "p.adjust") %in% colnames(df))) return(NULL)
    gobp_filter(df)
  }
  sig_ids_of <- function(df)
    if (is.null(df)) character(0) else unique(df$ID[!is.na(df$p.adjust) & df$p.adjust <= SIG_CUT])
  jaccard <- function(a, b) { u <- length(union(a, b)); if (u == 0) NA_real_ else round(length(intersect(a, b)) / u, 4) }
  
  classify_pair <- function(x, y) {
    j  <- merge(x, y, by = "ID", suffixes = c(".x", ".y"))
    xs <- j$p.adjust.x <= SIG_CUT; ys <- j$p.adjust.y <= SIG_CUT
    grp <- ifelse(xs & ys, "Both", ifelse(xs & !ys, "Merged", ifelse(!xs & ys, "Single", "None")))
    grp[is.na(grp)] <- "None"          
    j$`Sig. Group` <- grp
    j
  }
  rbind_fill <- function(lst) {
    lst <- Filter(function(d) !is.null(d) && nrow(d), lst)
    if (!length(lst)) return(NULL)
    cols <- unique(unlist(lapply(lst, names)))
    do.call(rbind, lapply(lst, function(d) {
      for (cn in setdiff(cols, names(d))) d[[cn]] <- NA
      d[, cols, drop = FALSE]
    }))
  }
  
  mf_relaxed <- list(
    gsea = load_relaxed(file.path(OUT_DIR, sprintf("%s_gsea_relaxed.tsv", TISSUE))),
    ora  = load_relaxed(file.path(OUT_DIR, sprintf("%s_ora_relaxed.tsv",  TISSUE))))
  mf_sig <- lapply(mf_relaxed, sig_ids_of)
  
  single_relaxed <- list(); single_sig <- list()
  for (s in STUDIES) for (par in c("gsea", "ora")) {
    d <- load_relaxed(single_relaxed_path(s$id, par))
    single_relaxed[[paste(s$id, par)]] <- d
    single_sig[[paste(s$id, par)]]     <- sig_ids_of(d)
  }
  single_union <- lapply(c(gsea = "gsea", ora = "ora"), function(par)
    unique(unlist(lapply(STUDIES, function(s) single_sig[[paste(s$id, par)]]))))
  
  approach_sig  <- list(model_free_de = list(gsea = mf_sig$gsea, ora = mf_sig$ora))
  approach_meta <- list(model_free_de = list(arm = "pooled", method = "model_free_de"))
  for (arm in ARMS) for (m in PANEL) {
    d <- load_relaxed(merged_relaxed_path(arm, m$method, m$paradigm)); if (is.null(d)) next
    key <- sprintf("%s__%s", arm, m$method)
    if (is.null(approach_sig[[key]])) approach_sig[[key]] <- list()
    approach_sig[[key]][[m$paradigm]] <- sig_ids_of(d)
    approach_meta[[key]] <- list(arm = arm, method = m$method)
  }
  panel_keys  <- setdiff(names(approach_sig), "model_free_de")
  panel_union <- lapply(c(gsea = "gsea", ora = "ora"), function(par)
    unique(unlist(lapply(panel_keys, function(k) approach_sig[[k]][[par]]))))
  
  mf_single_rows <- list(); mf_single_details <- list()
  for (par in c("gsea", "ora")) {
    mr <- mf_relaxed[[par]]; if (is.null(mr)) next
    for (s in STUDIES) {
      sr <- single_relaxed[[paste(s$id, par)]]; if (is.null(sr)) next
      j  <- classify_pair(mr, sr)
      g  <- table(factor(j$`Sig. Group`, levels = c("Both", "Merged", "Single", "None")))
      both <- as.integer(g["Both"]); mo <- as.integer(g["Merged"]); so <- as.integer(g["Single"])
      tot  <- length(single_sig[[paste(s$id, par)]])   
      mf_single_rows[[length(mf_single_rows) + 1]] <- data.frame(
        Method = "model_free_de", Paradigm = par, Arm = "pooled", RR_Mission = s$rr,
        Total_in_Single_Study = tot, Single_Study_Only = so, Single_and_Merged = both,
        Overlap_with_Merged_Total = both + mo, Merged_Study_Only = mo,
        Pct_Single_and_Merged = if (tot > 0) round(100 * both / tot, 1) else NA_real_,
        Namespace = "GO:BP", row.names = NULL, stringsAsFactors = FALSE)
      if (WRITE_MF_PAIR_DETAILS) {
        kc <- intersect(c("ID", "Description.x", "NES.x", "p.adjust.x", "NES.y", "p.adjust.y", "Sig. Group"),
                        colnames(j))
        jj <- j[j$`Sig. Group` != "None", kc, drop = FALSE]
        if (nrow(jj)) mf_single_details[[length(mf_single_details) + 1]] <- data.frame(
          Method = "model_free_de", Paradigm = par, Arm = "pooled", RR_Mission = s$rr,
          jj, Namespace = "GO:BP", row.names = NULL, check.names = FALSE, stringsAsFactors = FALSE)
      }
    }
  }
  if (length(mf_single_rows)) {
    mf_single_tab <- do.call(rbind, mf_single_rows)
    write.csv(mf_single_tab, file.path(OUT_DIR, sprintf("%s_mf_vs_single_summary.csv", TISSUE)), row.names = FALSE)
    for (par in c("gsea", "ora")) {
      sub <- mf_single_tab[mf_single_tab$Paradigm == par, ]
      if (nrow(sub)) cat(sprintf("  %-4s: mean recovery %.1f%%  over %d studies (Both/Total_in_Single)\n",
                                 par, mean(sub$Pct_Single_and_Merged, na.rm = TRUE), nrow(sub)))
    }
  } else cat("  no single-study relaxed files found", sep = "")
  if (WRITE_MF_PAIR_DETAILS) {
    det <- rbind_fill(mf_single_details)
    if (!is.null(det)) write.csv(det, file.path(OUT_DIR, sprintf("%s_mf_vs_single_details.csv", TISSUE)), row.names = FALSE)
  }
  
  # recovery leaderboard: every approach vs the single-study union
  per_study_recovery <- function(ids, par) {
    v <- vapply(STUDIES, function(s) {
      ss <- single_sig[[paste(s$id, par)]]
      if (length(ss) == 0) NA_real_ else length(intersect(ids, ss)) / length(ss)
    }, numeric(1))
    if (all(is.na(v))) NA_real_ else mean(v, na.rm = TRUE)
  }
  lb_rows <- list()
  for (key in names(approach_sig)) for (par in c("gsea", "ora")) {
    ids <- approach_sig[[key]][[par]]; if (is.null(ids)) next
    su  <- single_union[[par]]
    lb_rows[[length(lb_rows) + 1]] <- data.frame(
      Paradigm = par, Approach = key, Arm = approach_meta[[key]]$arm, Method = approach_meta[[key]]$method,
      n_sig = length(ids), n_single_union = length(su), n_shared_union = length(intersect(ids, su)),
      recovery_union         = if (length(su)) round(length(intersect(ids, su)) / length(su), 3) else NA_real_,
      mean_recovery_perstudy = round(per_study_recovery(ids, par), 3),
      jaccard_vs_single_union = jaccard(ids, su),
      row.names = NULL, stringsAsFactors = FALSE)
  }
  if (length(lb_rows)) {
    leaderboard <- do.call(rbind, lb_rows)
    leaderboard <- leaderboard[order(leaderboard$Paradigm, -leaderboard$recovery_union), ]
    write.csv(leaderboard, file.path(OUT_DIR, sprintf("%s_recovery_leaderboard.csv", TISSUE)), row.names = FALSE)
    for (par in c("gsea", "ora")) {
      sub <- leaderboard[leaderboard$Paradigm == par, ]
      if (!nrow(sub) || all(is.na(sub$recovery_union))) next
      rk <- which(sub$Approach == "model_free_de")
      cat(sprintf("  %-4s leaderboard (top of %d): model_free_de ranks %s\n",
                  par, nrow(sub), if (length(rk)) as.character(rk) else "NA"))
      print(utils::head(sub[, c("Approach", "Arm", "n_sig", "recovery_union",
                                "mean_recovery_perstudy", "jaccard_vs_single_union")], 6), row.names = FALSE)
    }
  } else cat("  no approaches with sig terms\n")
  
  # model-free vs each panel cell
  mk_overlap <- function(comparator_type, arm, method, par, comp) {
    mf <- mf_sig[[par]]
    data.frame(comparator_type = comparator_type, Paradigm = par, Arm = arm, Method = method,
               n_model_free = length(mf), n_comparator = length(comp),
               shared = length(intersect(mf, comp)),
               model_free_only = length(setdiff(mf, comp)),
               comparator_only = length(setdiff(comp, mf)),
               jaccard = jaccard(mf, comp), row.names = NULL, stringsAsFactors = FALSE)
  }
  mf_meth_rows <- list()
  for (key in panel_keys) for (par in c("gsea", "ora")) {
    comp <- approach_sig[[key]][[par]]; if (is.null(comp)) next
    mf_meth_rows[[length(mf_meth_rows) + 1]] <- mk_overlap(
      "panel_cell", approach_meta[[key]]$arm, approach_meta[[key]]$method, par, comp)
  }
  for (par in c("gsea", "ora"))
    mf_meth_rows[[length(mf_meth_rows) + 1]] <- mk_overlap("panel_union", "ALL", "ALL", par, panel_union[[par]])
  if (length(mf_meth_rows)) {
    mf_meth_tab <- do.call(rbind, mf_meth_rows)
    write.csv(mf_meth_tab, file.path(OUT_DIR, sprintf("%s_mf_vs_methods_overlap.csv", TISSUE)), row.names = FALSE)
    for (par in c("gsea", "ora")) {
      u <- mf_meth_tab[mf_meth_tab$comparator_type == "panel_union" & mf_meth_tab$Paradigm == par, ]
      if (nrow(u)) cat(sprintf("  %-4s: model-free %d terms vs panel-union %d -> shared %d (Jaccard=%.3f)\n",
                               par, u$n_model_free, u$n_comparator, u$shared, u$jaccard))
    }
  } else cat("  no panel cells found \n")
}
