# ============================================================
# 17_integration_diagnostics.R

# OUTPUTS  
#   d1_retention_summary.csv     per paradigm x method x arm x study  (DESCRIPTIVE
#                                full-merge retention; confounded by construction)
#   d1_denoising_loso_meta.csv   per study: confound-free LOSO-meta retention AUC
#   d1_singleonly_reliability.csv per study x single-only term: within-study
#                                stability + heterogeneity_loss/noise verdict
#   d2_meta_per_gene.csv         per-gene meta-analysis (z, p, Q, I2, sign-agree)
#   d2_meta_gsea_referee.tsv     relaxed meta-GSEA result (the referee universe)
#   d2_merged_only_verdict.csv   per merged-only term: meta-recovered? I2? LOSO?
#   d2_discovery_summary.csv     per GSEA cell: #merged-only, #recovered, frac
#
# ============================================================

suppressPackageStartupMessages({ library(dplyr) })

# ------------------------------------------------------------
# ------------------------------------------------------------
ROOT      <- "Path/to/liver"   
CMP_DIR   <- "data/go_bp_terms"
EXPR_DIR  <- "data/single_dge"
OUT_DIR   <- file.path(CMP_DIR, "integration_diagnostics")

SIG_CUT       <- 0.1            
ORA_NAMESPACE <- "GO:BP"        
D2_RANK_METRIC<- "meta_z"       # signed cross-study statistic ranking the referee
ORGANISM      <- "org.Mm.eg.db"
SEED          <- 12345
N_PERM        <- 2000           # label-permutation reps for the retention AUC null

# Which sub-analyses to run.
RUN_D1_MECHANICAL  <- TRUE      # descriptive full-merge retention (base R; confounded)
RUN_D1_DENOISING   <- TRUE      # confound-free LOSO-meta retention (needs Bioconductor)
RUN_D1_RELIABILITY <- TRUE     # within-study subsampling (SLOW: re-runs DESeq per study)
RUN_D2             <- FALSE       # merged-only discovery validation (needs Bioconductor)

WITHIN_B          <- 30         # subsamples per study (raise for a tighter estimate)
WITHIN_FRAC       <- 0.8        # fraction of each design cell kept per subsample
STABLE_FREQ       <- 0.6        # selection_freq >= this  -> heterogeneity_loss
NOISE_FREQ        <- 0.3        # selection_freq <  this  -> likely_noise
DGE_SCRIPT        <- "10_dge_singlestudy.R"   # builders sourced from here (define-only)

# A referee is needed by D1-DENOISING, D1-RELIABILITY (to define "single-only"),
# and D2
NEED_REFEREE <- RUN_D1_DENOISING || RUN_D1_RELIABILITY || RUN_D2

ARMS  <- c("zscore", "deseq2", "ruvg", "combat_ref")
PANEL <- list(
  list(method = "svm", paradigm = "gsea"), list(method = "glm", paradigm = "gsea"),
  list(method = "elasticnet", paradigm = "ora"), list(method = "xgboost", paradigm = "ora"),
  list(method = "svm", paradigm = "ora"), list(method = "glm", paradigm = "ora")
)

SF_LEVEL          <- "Space_Flight"
GC_LEVELS_DEFAULT <- c("Ground_Control", "Cohort_Control_1", "Cohort_Control_2")


STUDIES <- list(
  list(rr = "RR1_NASA", id = "168_RR1", expr = "glds168rr1_fc_and_pvals.csv",
       builder = "glds168_rr1_dds", bargs = list("168", "168"), gc_levels = NULL),
  list(rr = "RR6",      id = "245",     expr = "glds245_fc_and_pvals.csv",
       builder = "glds245_dds",     bargs = list("245", "245"), gc_levels = NULL),
  list(rr = "RR8",      id = "379",     expr = "glds379_fc_and_pvals.csv",
       builder = "glds379_dds",     bargs = list("379", "379"), gc_levels = NULL),
  list(rr = "RR9",      id = "242",     expr = "glds242_fc_and_pvals.csv",
       builder = "glds242_dds",     bargs = list("242", "242"),
       gc_levels = c("Cohort_Control_1")),
  list(rr = "RR3",      id = "168_RR3", expr = "glds168rr3_fc_and_pvals.csv",
       builder = "glds168_rr3_dds", bargs = list("168", "168"), gc_levels = NULL),
  list(rr = "RR47",     id = "47",      expr = "glds47_fc_and_pvals.csv",
       builder = "glds47_dds",      bargs = list("47", "47"),  gc_levels = NULL)
)

# ------------------------------------------------------------
# IO HELPERS 
# ------------------------------------------------------------
rd <- function(p) read.table(p, sep = "\t", header = TRUE, quote = "\"", stringsAsFactors = FALSE)

relx <- function(paradigm) if (paradigm == "gsea") "gseGO_high_pval" else "ora_high_pval"
merged_relaxed_path <- function(arm, method, paradigm)
  file.path(CMP_DIR, sprintf("%s_%s_%s.tsv", arm, method, relx(paradigm)))
single_relaxed_path <- function(id, paradigm)
  file.path(CMP_DIR, sprintf("%s_%s.tsv", id, relx(paradigm)))

load_relaxed <- function(path, paradigm) {
  if (!file.exists(path)) return(NULL)
  df <- rd(path)
  if (!nrow(df) || !all(c("ID", "p.adjust") %in% colnames(df))) return(NULL)
  if (paradigm == "ora" && !is.na(ORA_NAMESPACE) && "source" %in% colnames(df))
    df <- df[df$source == ORA_NAMESPACE, , drop = FALSE]
  df
}
sig_ids <- function(df) if (is.null(df)) character(0) else
  unique(df$ID[!is.na(df$p.adjust) & df$p.adjust <= SIG_CUT])

# ------------------------------------------------------------
# SHARED RETENTION STATISTICS 
# ------------------------------------------------------------
auc_from_scores <- function(score_pos, score_neg) {
  np <- length(score_pos); nn <- length(score_neg)
  if (np == 0 || nn == 0) return(NA_real_)
  r <- rank(c(score_pos, score_neg))                 # average ranks for ties
  (sum(r[seq_len(np)]) - np * (np + 1) / 2) / (np * nn)
}

# Label-permutation null for the AUC. 
auc_perm_p <- function(scores, is_pos, B = N_PERM, seed = SEED) {
  np <- sum(is_pos); nn <- sum(!is_pos); n <- length(scores)
  if (np == 0 || nn == 0) return(NA_real_)
  r   <- rank(scores)                                # ties handled once
  obs <- (sum(r[is_pos]) - np * (np + 1) / 2) / (np * nn)
  if (!is.finite(obs)) return(NA_real_)
  if (!is.null(seed)) set.seed(seed)
  ge <- 0L
  for (b in seq_len(B)) {
    s <- sum(r[sample.int(n, np)])
    a <- (s - np * (np + 1) / 2) / (np * nn)
    if (a >= obs - 1e-12) ge <- ge + 1L
  }
  (ge + 1) / (B + 1)                                 # one-sided, add-one
}


retention_stats <- function(single_df, recovered_ids, paradigm, has_nes,
                            n_perm = N_PERM) {
  ssig <- single_df[!is.na(single_df$p.adjust) & single_df$p.adjust <= SIG_CUT, , drop = FALSE]
  if (!nrow(ssig)) return(NULL)
  recovered <- ssig$ID %in% recovered_ids
  score     <- -log10(pmax(ssig$p.adjust, .Machine$double.xmin))   # single-study strength
  pos <- score[recovered]; neg <- score[!recovered]
  auc <- auc_from_scores(pos, neg)
  wp  <- if (length(pos) && length(neg))
    suppressWarnings(wilcox.test(pos, neg, alternative = "greater")$p.value) else NA_real_
  pp  <- auc_perm_p(score, recovered, B = n_perm)
  out <- data.frame(
    n_single_sig = nrow(ssig), n_recovered = sum(recovered), n_notrecovered = sum(!recovered),
    recovered_frac = round(mean(recovered), 3),
    AUC_by_signif = round(auc, 3),
    rank_biserial = round(2 * auc - 1, 3),
    wilcox_greater_p = signif(wp, 3),
    perm_p = signif(pp, 3),
    median_logp_recovered = round(median(pos), 3),
    median_logp_notrecovered = round(median(neg), 3),
    stringsAsFactors = FALSE)
  if (has_nes && "NES" %in% colnames(ssig)) {              
    a_nes <- abs(ssig$NES)
    out$AUC_by_absNES <- round(auc_from_scores(a_nes[recovered], a_nes[!recovered]), 3)
    out$median_absNES_recovered <- round(median(a_nes[recovered]), 3)
    out$median_absNES_notrecovered <- round(median(a_nes[!recovered]), 3)
  }
  out
}

# Console summary of an AUC column
summarize_auc <- function(tab, auc_col, group_col, note = "") {
  cat(sprintf("  %s by %s%s:\n", auc_col, group_col,
              if (nzchar(note)) paste0("  (", note, ")") else ""))
  for (g in unique(tab[[group_col]])) {
    sub <- tab[tab[[group_col]] == g, , drop = FALSE]
    v <- sub[[auc_col]]; w <- sub$n_single_sig
    ok <- is.finite(v) & is.finite(w)
    if (!any(ok)) { cat(sprintf("    %-12s  (no comparable strata)\n", g)); next }
    um <- mean(v[ok]); wm <- sum(v[ok] * w[ok]) / sum(w[ok])
    qs <- quantile(v[ok], c(0.25, 0.5, 0.75), names = FALSE)
    cat(sprintf("    %-12s unweighted=%.3f  n-weighted=%.3f  median=%.3f [IQR %.3f-%.3f]  (k=%d)\n",
                g, um, wm, qs[2], qs[1], qs[3], sum(ok)))
  }
}

# ------------------------------------------------------------
# D1-MECHANICAL
# ------------------------------------------------------------
to_legacy_retain <- function(df) {
  ren <- c(n_recovered = "n_retained", n_notrecovered = "n_dropped",
           recovered_frac = "retained_frac", AUC_by_signif = "AUC_retain_by_signif",
           median_logp_recovered = "median_logp_retained",
           median_logp_notrecovered = "median_logp_dropped",
           AUC_by_absNES = "AUC_retain_by_absNES",
           median_absNES_recovered = "median_absNES_retained",
           median_absNES_notrecovered = "median_absNES_dropped")
  nm <- colnames(df); hit <- nm %in% names(ren); nm[hit] <- ren[nm[hit]]
  colnames(df) <- nm; df
}

run_d1_mechanical <- function(single_cache) {
  rows <- list()
  for (m in PANEL) {
    meth <- m$method; par <- m$paradigm
    for (arm in ARMS) {
      merged <- load_relaxed(merged_relaxed_path(arm, meth, par), par); if (is.null(merged)) next
      msig <- sig_ids(merged)
      for (s in STUDIES) {
        sdf <- single_cache[[paste(s$id, par)]]; if (is.null(sdf)) next
        st <- retention_stats(sdf, msig, par, has_nes = (par == "gsea")); if (is.null(st)) next
        rows[[length(rows) + 1]] <- cbind(
          data.frame(Paradigm = par, Method = meth, Arm = arm, RR_Mission = s$rr,
                     comparator = "full_merge", stringsAsFactors = FALSE), st)
      }
    }
  }
  if (!length(rows)) { cat("no strata found\n"); return(invisible(NULL)) }
  tab <- dplyr::bind_rows(rows)
  tab$perm_p_adj <- signif(p.adjust(tab$perm_p, "BH"), 3)  
  tab <- to_legacy_retain(tab)
  write.csv(tab, file.path(OUT_DIR, "d1_retention_summary.csv"), row.names = FALSE)
  summarize_auc(tab, "AUC_retain_by_signif", "Paradigm", "DESCRIPTIVE; >0.5 partly mechanical")
  summarize_auc(tab, "AUC_retain_by_signif", "Arm",      "DESCRIPTIVE; compare to D1-DENOISING")
  invisible(tab)
}

# ------------------------------------------------------------
# META-ANALYSIS + meta-GSEA referee 
# ------------------------------------------------------------
read_study_de <- function(expr_csv) {
  p <- file.path(EXPR_DIR, expr_csv)
  if (!file.exists(p)) return(NULL)
  fc <- read.csv(p, header = TRUE, stringsAsFactors = FALSE)
  need <- c("gene", "log2fc", "lfcSE")
  if (!all(need %in% colnames(fc))) stop("expected ", paste(need, collapse = "/"), " in ", expr_csv)
  fc <- fc[!is.na(fc$log2fc) & !is.na(fc$lfcSE) & fc$lfcSE > 0, c("gene", "log2fc", "lfcSE")]
  fc[!duplicated(fc$gene), ]
}

meta_analyze <- function(de_list) {
  de_list <- Filter(Negate(is.null), de_list)
  genes <- sort(unique(unlist(lapply(de_list, function(d) d$gene))))
  k_studies <- length(de_list)
  Y <- matrix(NA_real_, length(genes), k_studies, dimnames = list(genes, names(de_list)))
  V <- Y
  for (j in seq_along(de_list)) {
    d <- de_list[[j]]; idx <- match(d$gene, genes)
    Y[idx, j] <- d$log2fc; V[idx, j] <- d$lfcSE^2
  }
  mask <- is.finite(Y) & is.finite(V) & V > 0
  Yc <- Y; Yc[!mask] <- 0
  W  <- ifelse(mask, 1 / V, 0)                       # fixed-effect weights
  k  <- rowSums(mask)
  sumW <- rowSums(W); sumWY <- rowSums(W * Yc); sumWY2 <- rowSums(W * Yc * Yc)
  sumW2 <- rowSums(W * W)
  Q  <- sumWY2 - (sumWY^2) / sumW                    # Cochran's Q
  df <- k - 1
  C  <- sumW - sumW2 / sumW
  tau2 <- pmax(0, (Q - df) / C); tau2[!is.finite(tau2)] <- 0    # DL between-study var
  Vt <- V + tau2                                     # tau2 (length=genes) recycled per row
  Wr <- ifelse(mask, 1 / Vt, 0)                      # random-effects weights
  sumWr <- rowSums(Wr)
  mu <- rowSums(Wr * Yc) / sumWr
  se <- sqrt(1 / sumWr)
  z  <- mu / se
  p  <- 2 * pnorm(-abs(z))
  I2 <- ifelse(Q > df & Q > 0, pmax(0, (Q - df) / Q), 0)
  posf <- rowSums(mask & Y > 0) / k; negf <- rowSums(mask & Y < 0) / k
  sign_agree <- pmax(posf, negf)
  out <- data.frame(gene = genes, k = k, meta_log2fc = mu, meta_se = se, meta_z = z,
                    meta_p = p, Q = Q, I2 = round(I2, 3), sign_agreement = round(sign_agree, 3),
                    row.names = NULL, stringsAsFactors = FALSE)
  out <- out[out$k >= 1 & is.finite(out$meta_z), ]
  out$meta_padj <- p.adjust(out$meta_p, "BH")
  out
}

# Bioconductor-dependent helpers 
ensembl_to_entrez_ranking <- function(ens_ids, scores) {
  df <- data.frame(ENSEMBL = ens_ids, score = scores, stringsAsFactors = FALSE)
  df$ENTREZID <- AnnotationDbi::mapIds(org.Mm.eg.db, keys = df$ENSEMBL,
                                       keytype = "ENSEMBL", column = "ENTREZID",
                                       multiVals = "first")
  df <- df[!is.na(df$ENTREZID) & !duplicated(df$ENTREZID), ]
  v <- df$score; names(v) <- df$ENTREZID
  sort(v, decreasing = TRUE)
}

gsego_relaxed <- function(ranking) {
  if (!is.null(SEED)) set.seed(SEED)
  gse <- clusterProfiler::gseGO(geneList = ranking, ont = "BP", keyType = "ENTREZID",
                                exponent = 1, eps = 0, pvalueCutoff = 0.9,
                                pAdjustMethod = "BH", OrgDb = org.Mm.eg.db,
                                seed = !is.null(SEED))
  as.data.frame(gse@result)
}

split_core <- function(x) if (is.na(x) || !nzchar(x)) character(0) else strsplit(x, "/", fixed = TRUE)[[1]]

# Build the meta-referee 
build_referee <- function() {
  suppressPackageStartupMessages({
    library(clusterProfiler); library(ORGANISM, character.only = TRUE); library(AnnotationDbi)
  })
  de_list <- setNames(lapply(STUDIES, function(s) read_study_de(s$expr)),
                      vapply(STUDIES, function(s) s$rr, character(1)))
  de_list <- Filter(Negate(is.null), de_list)
  if (length(de_list) < 2) stop("need >=2 single-study DE tables for a meta-analysis")
  meta <- meta_analyze(de_list)
  cat(sprintf("  meta-analysis over %d studies, %d genes (median I2 = %.2f)\n",
              length(de_list), nrow(meta), median(meta$I2, na.rm = TRUE)))
  
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
  loso <- setNames(lapply(seq_along(de_list),
                          function(k) referee_sig(meta_analyze(de_list[-k]))$sig),
                   names(de_list))
  cat(sprintf("  meta-GSEA referee: %d significant terms (%d LOSO folds)\n",
              length(ref_full$sig), length(loso)))
  list(de_list = de_list, meta = meta, het = het,
       ref_full = ref_full$relaxed, ref_full_sig = ref_full$sig, loso = loso)
}

# ------------------------------------------------------------
# D1-DENOISING -> confound-free LOSO-meta retention  
# ------------------------------------------------------------

run_d1_denoising <- function(referee, single_cache) {
  rows <- list()
  for (s in STUDIES) {
    if (!s$rr %in% names(referee$loso)) {
      cat(sprintf("  skip %s (no DE table in meta corpus)\n", s$rr)); next
    }
    sdf <- single_cache[[paste(s$id, "gsea")]]; if (is.null(sdf)) next
    st <- retention_stats(sdf, referee$loso[[s$rr]], "gsea", has_nes = TRUE); if (is.null(st)) next
    rows[[length(rows) + 1]] <- cbind(
      data.frame(Paradigm = "gsea", RR_Mission = s$rr,
                 comparator = "loso_meta_excl_k", stringsAsFactors = FALSE), st)
  }
  if (!length(rows)) { cat("  no studies with single GSEA terms + a LOSO referee\n"); return(invisible(NULL)) }
  tab <- dplyr::bind_rows(rows)
  tab$perm_p_adj <- signif(p.adjust(tab$perm_p, "BH"), 3)
  write.csv(tab, file.path(OUT_DIR, "d1_denoising_loso_meta.csv"), row.names = FALSE)
  summarize_auc(tab, "AUC_by_signif", "Paradigm", "confound-free; >0.5 = real denoising")
  for (i in seq_len(nrow(tab)))
    cat(sprintf("    %-9s AUC=%.3f  recovered %d/%d  perm_p=%s (adj %s)\n",
                tab$RR_Mission[i], tab$AUC_by_signif[i], tab$n_recovered[i],
                tab$n_single_sig[i], format(tab$perm_p[i]), format(tab$perm_p_adj[i])))
  invisible(tab)
}

# ------------------------------------------------------------
# D1-RELIABILITY -> within-study reproducibility of single-only terms
# ------------------------------------------------------------

source_definitions <- function(path, envir) {
  if (!file.exists(path)) stop("DGE_SCRIPT not found: ", path,
                               " (needed for the within-study reliability check)")
  keep_fn <- c("<-", "=", "assign", "library", "require",
               "suppressPackageStartupMessages", "setClass", "setGeneric",
               "setMethod", "setRefClass")
  for (e in parse(path)) {
    if (is.call(e) && !(as.character(e[[1L]])[1L] %in% keep_fn)) next   # skip bare drivers
    eval(e, envir = envir)
  }
  invisible(NULL)
}

within_study_term_stability <- function(study, builders_env,
                                        B = WITHIN_B, frac = WITHIN_FRAC, seed = SEED) {
  suppressPackageStartupMessages({
    library(DESeq2); library(clusterProfiler); library(ORGANISM, character.only = TRUE)
  })
  make_dds <- get(study$builder, envir = builders_env)
  raw <- do.call(make_dds, study$bargs)$data            
  gc_levels <- if (is.null(study$gc_levels)) GC_LEVELS_DEFAULT else study$gc_levels
  
  keep0 <- as.character(raw$spaceflight) %in% c(SF_LEVEL, gc_levels)
  raw <- raw[, keep0]
  design0 <- DESeq2::design(raw)
  
  cd0     <- as.data.frame(SummarizedExperiment::colData(raw))
  dvars   <- intersect(all.vars(design0), colnames(cd0))
  dvars   <- dvars[vapply(dvars, function(v) is.factor(cd0[[v]]) || is.character(cd0[[v]]), logical(1))]
  if (!length(dvars)) dvars <- "spaceflight"
  strata_cols <- lapply(dvars, function(v) as.character(cd0[[v]]))
  strata  <- if (length(strata_cols) == 1L) factor(strata_cols[[1L]])
  else do.call(interaction, c(strata_cols, list(drop = TRUE)))
  
  if (!is.null(seed)) set.seed(seed)
  freq <- list(); n_ok <- 0L
  for (b in seq_len(B)) {
    keep <- unlist(lapply(split(seq_len(ncol(raw)), strata), function(ix) {
      n <- length(ix); if (n <= 1) return(ix)         
      sample(ix, max(2L, floor(n * frac)))
    }))
    sub <- raw[, sort(keep)]
    
    pooled <- ifelse(as.character(sub$spaceflight) == SF_LEVEL, "Space_Flight", "Ground_Control")
    sub$spaceflight <- factor(pooled, levels = c("Ground_Control", "Space_Flight"))
    for (cn in colnames(SummarizedExperiment::colData(sub))) {
      col <- SummarizedExperiment::colData(sub)[[cn]]
      if (is.factor(col)) SummarizedExperiment::colData(sub)[[cn]] <- droplevels(col)
    }
    if (nlevels(sub$spaceflight) < 2) next
    sub$spaceflight <- relevel(sub$spaceflight, ref = "Ground_Control")
    design(sub) <- design0                             
    
    fit <- tryCatch(DESeq2::DESeq(sub, quiet = TRUE), error = function(e) NULL)
    if (is.null(fit)) next                           
    res <- tryCatch(DESeq2::results(fit, contrast = c("spaceflight", "Space_Flight", "Ground_Control")),
                    error = function(e) NULL)
    if (is.null(res)) next
    fc <- data.frame(gene = rownames(res), log2fc = res$log2FoldChange, stringsAsFactors = FALSE)
    fc <- fc[!is.na(fc$log2fc), ]
    rk  <- ensembl_to_entrez_ranking(fc$gene, fc$log2fc) 
    sig <- gsego_relaxed(rk)
    ids <- sig$ID[!is.na(sig$p.adjust) & sig$p.adjust <= SIG_CUT]
    for (id in ids) freq[[id]] <- (if (is.null(freq[[id]])) 0L else freq[[id]]) + 1L
    n_ok <- n_ok + 1L
  }
  freq_df <- if (length(freq))
    data.frame(ID = names(freq), selection_freq = round(unlist(freq) / max(1L, n_ok), 3),
               row.names = NULL, stringsAsFactors = FALSE)
  else data.frame(ID = character(0), selection_freq = numeric(0), stringsAsFactors = FALSE)
  list(freq = freq_df, n_success = n_ok, n_attempt = B)   # n_success distinguishes ran-but-empty from not-run
}

run_d1_reliability <- function(referee, single_cache) {
  builders_env <- new.env(parent = globalenv())
  source_definitions(DGE_SCRIPT, builders_env)
  
  rows <- list()
  for (s in STUDIES) {
    if (!s$rr %in% names(referee$loso)) { cat(sprintf("  skip %s (no LOSO referee)\n", s$rr)); next }
    sdf <- single_cache[[paste(s$id, "gsea")]]; if (is.null(sdf)) next
    ssig <- sdf[!is.na(sdf$p.adjust) & sdf$p.adjust <= SIG_CUT, , drop = FALSE]
    if (!nrow(ssig)) next
    single_only <- !(ssig$ID %in% referee$loso[[s$rr]])
    if (!any(single_only)) { cat(sprintf("  %-9s no single-only terms\n", s$rr)); next }
    
    stab <- tryCatch(within_study_term_stability(s, builders_env),
                     error = function(e) { cat(sprintf("  %-9s reliability failed: %s\n",
                                                       s$rr, conditionMessage(e))); NULL })
    nsucc   <- if (is.null(stab)) NA_integer_ else stab$n_success
    has_run <- !is.null(stab) && !is.na(nsucc) && nsucc > 0
    so <- ssig[single_only, , drop = FALSE]
    if (has_run) {
      sf <- stab$freq$selection_freq[match(so$ID, stab$freq$ID)]
      sf[is.na(sf)] <- 0                                 
    } else sf <- rep(NA_real_, nrow(so))                  
    
    verdict <- if (!has_run) rep("stability_not_run", nrow(so)) else
      ifelse(sf >= STABLE_FREQ, "heterogeneity_loss",
             ifelse(sf <  NOISE_FREQ,  "likely_noise", "ambiguous"))
    rows[[length(rows) + 1]] <- data.frame(
      RR_Mission = s$rr, ID = so$ID,
      Description = if ("Description" %in% colnames(so)) so$Description else NA,
      single_logp = round(-log10(pmax(so$p.adjust, .Machine$double.xmin)), 3),
      single_NES = if ("NES" %in% colnames(so)) round(so$NES, 3) else NA_real_,
      in_full_referee = so$ID %in% referee$ref_full_sig,
      within_study_selection_freq = sf, n_success = nsucc, n_attempt = WITHIN_B,
      verdict = verdict, row.names = NULL, stringsAsFactors = FALSE)
    cat(sprintf("  %-9s single-only=%d  het_loss=%d  noise=%d  ambiguous=%d  (subsamples ok: %s)\n",
                s$rr, nrow(so), sum(verdict == "heterogeneity_loss"),
                sum(verdict == "likely_noise"), sum(verdict == "ambiguous"),
                if (is.na(nsucc)) "0" else as.character(nsucc)))
  }
  if (length(rows)) {
    v <- dplyr::bind_rows(rows)
    write.csv(v, file.path(OUT_DIR, "d1_singleonly_reliability.csv"), row.names = FALSE)
  } else cat("  nothing to write\n")
  invisible(NULL)
}

# ------------------------------------------------------------
# DESIDERATUM 2: are merged-only terms genuine cross-study discoveries?
# ------------------------------------------------------------
run_desideratum2 <- function(referee) {
  write.csv(referee$meta, file.path(OUT_DIR, "d2_meta_per_gene.csv"), row.names = FALSE)
  write.table(referee$ref_full, file.path(OUT_DIR, "d2_meta_gsea_referee.tsv"),
              sep = "\t", quote = TRUE, row.names = FALSE)
  het <- referee$het; loso <- referee$loso; n_loso <- length(loso)
  
  single_gsea_union <- unique(unlist(lapply(STUDIES, function(s)
    sig_ids(load_relaxed(single_relaxed_path(s$id, "gsea"), "gsea")))))
  
  verdict_rows <- list(); summ_rows <- list()
  for (arm in ARMS) for (m in PANEL) {
    if (m$paradigm != "gsea") next                     
    merged <- load_relaxed(merged_relaxed_path(arm, m$method, "gsea"), "gsea"); if (is.null(merged)) next
    msig <- sig_ids(merged)
    merged_only <- setdiff(msig, single_gsea_union)    
    if (!length(merged_only)) next
    for (id in merged_only) {
      row <- merged[merged$ID == id, , drop = FALSE][1, ]
      core <- split_core(if ("core_enrichment" %in% colnames(row)) row$core_enrichment else NA)
      h <- het[intersect(core, rownames(het)), , drop = FALSE]
      recovered <- id %in% referee$ref_full_sig
      verdict_rows[[length(verdict_rows) + 1]] <- data.frame(
        Arm = arm, Method = m$method, ID = id,
        Description = if ("Description" %in% colnames(row)) row$Description else NA,
        merged_NES = round(row$NES, 3),
        meta_recovered = recovered,
        loso_survival = if (recovered) sum(vapply(loso, function(s) id %in% s, logical(1))) else 0L,
        n_loso = n_loso,
        n_leadingedge_in_meta = nrow(h),
        median_I2_leadingedge = if (nrow(h)) round(median(h$I2), 3) else NA_real_,
        median_signagree_leadingedge = if (nrow(h)) round(median(h$sign_agreement), 3) else NA_real_,
        row.names = NULL, stringsAsFactors = FALSE)
    }
    nrec <- sum(merged_only %in% referee$ref_full_sig)
    summ_rows[[length(summ_rows) + 1]] <- data.frame(
      Arm = arm, Method = m$method, n_merged_only = length(merged_only),
      n_meta_recovered = nrec, frac_recovered = round(nrec / length(merged_only), 3),
      row.names = NULL, stringsAsFactors = FALSE)
  }
  if (length(verdict_rows)) {
    v <- dplyr::bind_rows(verdict_rows)
    v$verdict <- ifelse(!v$meta_recovered, "artifact_candidate",
                        ifelse(v$loso_survival >= v$n_loso - 1 &
                                 (is.na(v$median_I2_leadingedge) | v$median_I2_leadingedge <= 0.5),
                               "credible_discovery", "fragile"))
    write.csv(v, file.path(OUT_DIR, "d2_merged_only_verdict.csv"), row.names = FALSE)
  }
  if (length(summ_rows)) {
    s <- dplyr::bind_rows(summ_rows)
    write.csv(s, file.path(OUT_DIR, "d2_discovery_summary.csv"), row.names = FALSE)
    cat(sprintf("  merged-only terms across GSEA cells: %d  | meta-recovered: %d (%.1f%%)\n",
                sum(s$n_merged_only), sum(s$n_meta_recovered),
                100 * sum(s$n_meta_recovered) / max(1, sum(s$n_merged_only))))
  }
  invisible(NULL)
}

# -----------------------------------------------------------
if (!exists("DIAG_TEST_MODE")) {
  setwd(ROOT)
  dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
  
  single_cache <- list()
  for (s in STUDIES) for (par in c("gsea", "ora"))
    single_cache[[paste(s$id, par)]] <- load_relaxed(single_relaxed_path(s$id, par), par)
  
  if (RUN_D1_MECHANICAL) run_d1_mechanical(single_cache)
  
  referee <- NULL
  if (NEED_REFEREE) {
    cat("\n== Building shared meta-GSEA referee ==\n")
    referee <- build_referee()
  }
  if (RUN_D1_DENOISING)   run_d1_denoising(referee, single_cache)
  if (RUN_D1_RELIABILITY) run_d1_reliability(referee, single_cache)
  if (RUN_D2)             run_desideratum2(referee)

}