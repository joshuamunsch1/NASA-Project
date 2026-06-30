# ============================================================
# 13_go_overlap_methodssecond_merged.R  
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
})

setwd("path/to/heart")

CMP_DIR <- "data/go_bp_terms"
SIG_DIR <- "data/go_terms_single_datasets"
OUT_DIR <- file.path(CMP_DIR, "method_overlap")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

SIG_CUT <- 0.1
WRITE_PAIR_DETAILS <- TRUE      

rd <- function(p) read.table(p, sep = "\t", header = TRUE, quote = "\"", stringsAsFactors = FALSE)

rbind_fill <- function(lst) {
  lst <- Filter(function(d) !is.null(d) && nrow(d), lst)
  if (!length(lst)) return(NULL)
  cols <- unique(unlist(lapply(lst, names)))
  do.call(rbind, lapply(lst, function(d) {
    for (cn in setdiff(cols, names(d))) d[[cn]] <- NA
    d[, cols, drop = FALSE]
  }))
}

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ARMS <- c("zscore", "deseq2", "ruvg", "combat_ref")

PANEL <- list(
  list(method = "svm",        paradigm = "gsea"),
  list(method = "svm",        paradigm = "ora"),
  list(method = "glm",        paradigm = "gsea"),
  list(method = "glm",        paradigm = "ora"),
  list(method = "elasticnet", paradigm = "gsea"),
  list(method = "elasticnet", paradigm = "ora"),
  list(method = "xgboost",    paradigm = "gsea"),
  list(method = "xgboost",    paradigm = "ora"),
)
STUDIES <- list(
  list(rr = "270", id = "270", sig_gsea = "GLDS-270-GO.tsv", sig_ora = "GLDS-270-ORA.tsv"),
  list(rr = "580", id = "580", sig_gsea = "GLDS-580-GO.tsv", sig_ora = "GLDS-580-ORA.tsv"),
  list(rr = "599", id = "599", sig_gsea = "GLDS-599-GO.tsv", sig_ora = "GLDS-599-ORA.tsv")
)

relx <- function(paradigm) if (paradigm == "gsea") "gseGO_high_pval" else "ora_high_pval"

merged_relaxed_path <- function(arm, method, paradigm)
  file.path(CMP_DIR, sprintf("%s_%s_%s.tsv", arm, method, relx(paradigm)))
single_relaxed_path <- function(id, paradigm)
  file.path(CMP_DIR, sprintf("%s_%s.tsv", id, relx(paradigm)))

cell_key <- function(arm, method, paradigm) sprintf("%s__%s__%s", arm, method, paradigm)

load_relaxed <- function(path) {
  if (!file.exists(path)) return(NULL)
  df <- rd(path)
  if (!nrow(df) || !all(c("ID", "p.adjust") %in% colnames(df))) return(NULL)
  df
}
sig_ids_of <- function(df) unique(df$ID[!is.na(df$p.adjust) & df$p.adjust <= SIG_CUT])

classify_pair <- function(x, y, only_x, only_y) {
  dplyr::inner_join(x, y, by = "ID", suffix = c(".x", ".y")) %>%
    dplyr::mutate(
      x_significant = p.adjust.x <= SIG_CUT,
      y_significant = p.adjust.y <= SIG_CUT,
      `Sig. Group` = dplyr::case_when(
        x_significant &  y_significant ~ "Both",
        x_significant & !y_significant ~ only_x,
        !x_significant &  y_significant ~ only_y,
        TRUE                           ~ "None"))
}
jaccard <- function(a, b) { u <- length(union(a, b)); if (u == 0) NA_real_ else round(length(intersect(a, b)) / u, 4) }

# ------------------------------------------------------------
# Load 
# ------------------------------------------------------------
cell_sig  <- list()   
cell_meta <- list()
desc_lookup <- data.frame(ID = character(), Description = character(), stringsAsFactors = FALSE)
for (arm in ARMS) for (m in PANEL) {
  df <- load_relaxed(merged_relaxed_path(arm, m$method, m$paradigm))
  if (is.null(df)) next
  cell <- cell_key(arm, m$method, m$paradigm)
  cell_sig[[cell]]  <- sig_ids_of(df)
  cell_meta[[cell]] <- list(arm = arm, method = m$method, paradigm = m$paradigm)
  if ("Description" %in% colnames(df))
    desc_lookup <- rbind(desc_lookup, df[!duplicated(df$ID), c("ID", "Description")])
}
desc_lookup <- desc_lookup[!duplicated(desc_lookup$ID), , drop = FALSE]
get_desc <- function(ids) desc_lookup$Description[match(ids, desc_lookup$ID)]
cells <- names(cell_sig)
if (length(cells) < 1) stop("No panel relaxed files found under ", CMP_DIR,
                            " -- run 11_gsea.R and 11b_ora.R first.")

# ============================================================
# BETWEEN-ARMS convergence
# ============================================================
presence_rows <- list()   
jaccard_rows  <- list()  
for (m in PANEL) {
  meth <- m$method; par <- m$paradigm
  present <- ARMS[vapply(ARMS, function(a) !is.null(cell_sig[[cell_key(a, meth, par)]]), logical(1))]
  if (length(present) < 1) { cat(sprintf("  [%-10s %s] no files -> skip\n", meth, par)); next }
  sig <- lapply(present, function(a) cell_sig[[cell_key(a, meth, par)]])
  names(sig) <- present
  all_ids <- sort(unique(unlist(sig)))
  if (!length(all_ids)) { cat(sprintf("  [%-10s %s] no significant terms -> skip\n", meth, par)); next }
  
  pres <- sapply(present, function(a) as.integer(all_ids %in% sig[[a]]))
  pres <- matrix(pres, nrow = length(all_ids), dimnames = list(all_ids, present))
  npt  <- rowSums(pres)
  k    <- length(present)
  
  pres_full <- matrix(NA_integer_, nrow = length(all_ids), ncol = length(ARMS),
                      dimnames = list(NULL, ARMS))
  pres_full[, present] <- pres
  presence_rows[[paste(meth, par)]] <- data.frame(
    Method = meth, Paradigm = par, ID = all_ids, Description = get_desc(all_ids),
    pres_full, n_arms = npt, is_core = (npt == k),
    row.names = NULL, check.names = FALSE, stringsAsFactors = FALSE)
  
  jm <- matrix(0, k, k, dimnames = list(present, present))
  for (i in seq_len(k)) for (j in seq_len(k)) jm[i, j] <- jaccard(sig[[present[i]]], sig[[present[j]]])
  for (i in seq_len(k)) for (j in seq_len(k))
    jaccard_rows[[length(jaccard_rows) + 1]] <- data.frame(
      Method = meth, Paradigm = par, arm_a = present[i], arm_b = present[j], jaccard = jm[i, j],
      row.names = NULL, stringsAsFactors = FALSE)
  
  core <- all_ids[npt == k]
  cat(sprintf("  [%-10s %s] arms=%d  union=%d  core(all arms)=%d\n",
              meth, par, k, length(all_ids), length(core)))
}
if (length(presence_rows))
  write.csv(do.call(rbind, presence_rows),
            file.path(OUT_DIR, "between_arms_presence.csv"), row.names = FALSE)
if (length(jaccard_rows))
  write.csv(do.call(rbind, jaccard_rows),
            file.path(OUT_DIR, "between_arms_jaccard.csv"), row.names = FALSE)

# ============================================================
#  MERGED vs SINGLE, per (method, paradigm)
# ============================================================
single_rel <- list(); single_tot <- list()
for (s in STUDIES) for (par in c("gsea", "ora")) {
  rp <- single_relaxed_path(s$id, par)
  single_rel[[paste(s$id, par)]] <- if (file.exists(rp)) rd(rp) else NULL
  sp <- file.path(SIG_DIR, if (par == "gsea") s$sig_gsea else s$sig_ora)
  single_tot[[paste(s$id, par)]] <- if (file.exists(sp)) nrow(rd(sp)) else NA_integer_
}

summary_rows <- list()   
detail_rows  <- list()   
for (m in PANEL) {
  meth <- m$method; par <- m$paradigm; long <- list()
  for (arm in ARMS) {
    merged <- load_relaxed(merged_relaxed_path(arm, meth, par)); if (is.null(merged)) next
    for (s in STUDIES) {
      sr <- single_rel[[paste(s$id, par)]]; if (is.null(sr)) next
      joined <- classify_pair(merged, sr, "Merged", "Single")
      if (WRITE_PAIR_DETAILS) {
        kc <- intersect(c("ID","Description.x","NES.x","p.adjust.x","NES.y","p.adjust.y","Sig. Group"), colnames(joined))
        if (length(kc) && nrow(joined))
          detail_rows[[length(detail_rows) + 1]] <- data.frame(
            Method = meth, Paradigm = par, Arm = arm, RR_Mission = s$rr,
            joined[, kc, drop = FALSE],
            row.names = NULL, check.names = FALSE, stringsAsFactors = FALSE)
      }
      g <- table(factor(joined$`Sig. Group`, levels = c("Both","Merged","Single","None")))
      both <- as.integer(g["Both"]); mo <- as.integer(g["Merged"]); so <- as.integer(g["Single"])
      tot <- single_tot[[paste(s$id, par)]]; if (is.na(tot)) tot <- both + so
      long[[length(long)+1]] <- data.frame(
        Method = meth, Paradigm = par, Arm = arm, RR_Mission = s$rr,
        Total_in_Single_Study = as.integer(tot), Single_Study_Only = so,
        Single_and_Merged = both, Overlap_with_Merged_Total = both + mo,
        Merged_Study_Only = mo,
        Pct_Single_and_Merged = if (tot > 0) round(100*both/tot, 1) else NA_real_,
        row.names = NULL, stringsAsFactors = FALSE)
    }
  }
  if (length(long)) {
    tab <- do.call(rbind, long)
    summary_rows[[paste(meth, par)]] <- tab
    cat(sprintf("  [%-10s %s] %d arm x study rows  (mean both%%=%.1f)\n",
                meth, par, nrow(tab), mean(tab$Pct_Single_and_Merged, na.rm = TRUE)))
  } else cat(sprintf("  [%-10s %s] no pairs -> skip\n", meth, par))
}
if (length(summary_rows))
  write.csv(do.call(rbind, summary_rows),
            file.path(OUT_DIR, "merged_vs_single_summary.csv"), row.names = FALSE)
if (WRITE_PAIR_DETAILS) {
  for (i in 1:length(detail_rows)) {
    detail_rows[[i]] <- detail_rows[[i]][detail_rows[[i]]$`Sig. Group` != "None",]
  }
  det <- rbind_fill(detail_rows)
  if (!is.null(det))
    write.csv(det, file.path(OUT_DIR, "merged_vs_single_details.csv"), row.names = FALSE)
}

# ============================================================
# PANEL CONVERGENCE 
# ============================================================
all_ids <- sort(unique(unlist(cell_sig)))
pres <- sapply(cells, function(c) as.integer(all_ids %in% cell_sig[[c]]))
pres <- matrix(pres, nrow = length(all_ids), dimnames = list(all_ids, cells))
is_gsea <- vapply(cells, function(c) cell_meta[[c]]$paradigm == "gsea", logical(1))
n_gsea  <- rowSums(pres[, is_gsea,  drop = FALSE])
n_ora   <- rowSums(pres[, !is_gsea, drop = FALSE])
n_cells <- rowSums(pres)

presence_df <- data.frame(ID = all_ids, Description = get_desc(all_ids), pres,
                          n_cells = n_cells, n_gsea_cells = n_gsea, n_ora_cells = n_ora,
                          row.names = NULL, check.names = FALSE)
presence_df <- presence_df[order(-presence_df$n_cells, presence_df$ID), ]
write.csv(presence_df, file.path(OUT_DIR, "panel_presence_matrix.csv"), row.names = FALSE)

shared_by_k <- as.data.frame(table(factor(n_cells, levels = seq_len(length(cells)))))
colnames(shared_by_k) <- c("n_cells", "n_terms")
write.csv(shared_by_k, file.path(OUT_DIR, "panel_shared_by_k.csv"), row.names = FALSE)

cell_summary <- do.call(rbind, lapply(cells, function(c) data.frame(
  cell = c, arm = cell_meta[[c]]$arm, method = cell_meta[[c]]$method,
  paradigm = cell_meta[[c]]$paradigm,
  n_significant = length(cell_sig[[c]]),
  n_unique_in_panel = sum(n_cells == 1 & pres[, c] == 1),
  row.names = NULL, stringsAsFactors = FALSE)))
write.csv(cell_summary, file.path(OUT_DIR, "panel_cell_summary.csv"), row.names = FALSE)

# within-paradigm convergence (core = significant in EVERY cell of that paradigm)
wp <- list()
for (par in c("gsea", "ora")) {
  pc <- cells[vapply(cells, function(c) cell_meta[[c]]$paradigm == par, logical(1))]
  if (!length(pc)) next
  sub <- pres[, pc, drop = FALSE]; np <- rowSums(sub)
  uni <- all_ids[np >= 1]; core <- all_ids[np == length(pc)]
  wp[[par]] <- data.frame(paradigm = par, n_cells = length(pc),
                          union_significant = length(uni), core_all_cells = length(core),
                          median_support = median(np[np >= 1]),
                          row.names = NULL, stringsAsFactors = FALSE)
}
if (length(wp)) write.csv(do.call(rbind, wp),
                          file.path(OUT_DIR, "convergence_within_paradigm.csv"), row.names = FALSE)
cat(sprintf("  pooled union=%d  shared-by-all-%d-cells=%d\n",
            length(all_ids), length(cells), sum(n_cells == length(cells))))
print(shared_by_k)

# ============================================================
# CROSS-PARADIGM COMPLEMENTARITY  (GSEA-union vs ORA-union)
# ============================================================
gsea_cells <- cells[is_gsea]; ora_cells <- cells[!is_gsea]
gsea_union <- unique(unlist(cell_sig[gsea_cells]))
ora_union  <- unique(unlist(cell_sig[ora_cells]))
shared   <- intersect(gsea_union, ora_union)
gsea_only<- setdiff(gsea_union, ora_union)
ora_only <- setdiff(ora_union,  gsea_union)

comp_rows <- list(data.frame(
  scope = "ALL_ARMS", n_gsea = length(gsea_union), n_ora = length(ora_union),
  shared = length(shared), gsea_only = length(gsea_only), ora_only = length(ora_only),
  jaccard = jaccard(gsea_union, ora_union), row.names = NULL, stringsAsFactors = FALSE))

for (arm in ARMS) {
  gc <- cells[is_gsea  & vapply(cells, function(c) cell_meta[[c]]$arm == arm, logical(1))]
  oc <- cells[!is_gsea & vapply(cells, function(c) cell_meta[[c]]$arm == arm, logical(1))]
  if (!length(gc) || !length(oc)) next
  gu <- unique(unlist(cell_sig[gc])); ou <- unique(unlist(cell_sig[oc]))
  comp_rows[[length(comp_rows)+1]] <- data.frame(
    scope = arm, n_gsea = length(gu), n_ora = length(ou),
    shared = length(intersect(gu, ou)), gsea_only = length(setdiff(gu, ou)),
    ora_only = length(setdiff(ou, gu)), jaccard = jaccard(gu, ou),
    row.names = NULL, stringsAsFactors = FALSE)
}
write.csv(do.call(rbind, comp_rows), file.path(OUT_DIR, "paradigm_complementarity.csv"), row.names = FALSE)


support_in <- function(ids, cset) vapply(ids, function(i) sum(vapply(cset, function(c) i %in% cell_sig[[c]], logical(1))), integer(1))
mk_terms <- function(cls, ids, gsea_n, ora_n) {       # length-safe even for empty classes
  n <- length(ids)
  data.frame(class = rep(cls, n), ID = ids, Description = get_desc(ids),
             gsea_cells = rep(gsea_n, length.out = n),
             ora_cells  = rep(ora_n,  length.out = n),
             row.names = NULL, stringsAsFactors = FALSE)
}
paradigm_terms <- rbind(
  mk_terms("shared",    shared,    support_in(shared, gsea_cells),    support_in(shared, ora_cells)),
  mk_terms("gsea_only", gsea_only, support_in(gsea_only, gsea_cells), NA_integer_),
  mk_terms("ora_only",  ora_only,  NA_integer_,                       support_in(ora_only, ora_cells)))
write.csv(paradigm_terms, file.path(OUT_DIR, "paradigm_terms.csv"), row.names = FALSE)


