# ============================================================
# 06_diagnostics.R
# Outputs:
#   data/diagnostics_pca.png
#   data/diagnostics_metrics.csv
# ============================================================

TISSUE       <- "heart"                                  # "liver" or "heart" 
PROJECT_ROOT   <- if (TISSUE == "liver") {
  "path/to/liver"
} else {
  "path/to/heart"
}

suppressPackageStartupMessages({
  library(edgeR)
  library(ggplot2)
  library(patchwork)
  library(cluster)
})

# ------------------------------------------------------------
# LOAD ORIGINAL + ALL CORRECTED MATRICES
# ------------------------------------------------------------
counts <- readRDS("pipeline/merged_counts.rds")
meta   <- readRDS("pipeline/meta.rds")

dge        <- DGEList(counts = counts)
dge        <- calcNormFactors(dge, method = "TMM")
raw_logcpm <- cpm(dge, log = TRUE, prior.count = 1)

maybe_load <- function(p){ 
  if (file.exists(p)){ 
    obj <- read.csv(p)
  }
  else {
    obj <- NULL
  }
  
  if (is.data.frame(obj)) {
    if (ncol(obj) > 0 && !is.numeric(obj[[1]])) {
      rn  <- as.character(obj[[1]])
      obj <- obj[, -1, drop = FALSE]
      rownames(obj) <- rn
    }
    obj <- as.matrix(obj)
  }
  
  storage.mode(obj) <- "double"
  obj
  
}

# log-transform if a matrix is still on count scale.
log_if_counts <- function(m){
  if (is.null(m)){
    NULL
  }
  else log2(m + 1)
}

methods <- list(
  "Raw (log2-CPM)" = raw_logcpm,
  "ComBat-ref"     = log_if_counts(maybe_load("corrected_combat_ref.csv")),
  "RUVg"           = log_if_counts(maybe_load("corrected_ruvg_counts.csv")),
  "DESeq2"         = maybe_load("corrected_deseq2.csv"),
  "zscore" = maybe_load('concat_df.csv')
)
methods <- methods[!vapply(methods, is.null, logical(1))]
cat("Comparing", length(methods), "matrices:",
    paste(names(methods), collapse = ", "), "\n")

# ------------------------------------------------------------
# PCA HELPER
# ------------------------------------------------------------
make_pca_df <- function(mat, meta) {
  v   <- apply(mat, 1, var, na.rm = TRUE)
  mat <- mat[is.finite(v) & v > 0, ]
  pca <- prcomp(t(mat), scale. = TRUE, center = TRUE)
  ve  <- (pca$sdev^2) / sum(pca$sdev^2)
  data.frame(
    PC1       = pca$x[, 1],
    PC2       = pca$x[, 2],
    study     = meta$study,
    condition = meta$condition,
    pc1_var   = ve[1],
    pc2_var   = ve[2]
  )
}

plot_one <- function(name, mat) {
  df <- make_pca_df(mat, meta)
  ggplot(df, aes(PC1, PC2, color = study, shape = condition)) +
    geom_point(size = 2.5, alpha = 0.85) +
    labs(
      title    = name,
      subtitle = sprintf("PC1 %.1f%% / PC2 %.1f%%",
                         100 * df$pc1_var[1], 100 * df$pc2_var[1])
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "right")
}

plots    <- Map(plot_one, names(methods), methods)
n_cols   <- 3
n_rows   <- ceiling(length(plots) / n_cols)
combined <- wrap_plots(plots, ncol = n_cols, guides = "collect")
ggsave("data/diagnostics_pca.png", combined,
       width = 5.3 * n_cols, height = 3.6 * n_rows, dpi = 150)

# ------------------------------------------------------------
# QUANTITATIVE METRICS  
# ------------------------------------------------------------
metrics <- lapply(names(methods), function(name) {
  mat <- methods[[name]]
  v   <- apply(mat, 1, var, na.rm = TRUE)
  mat <- mat[is.finite(v) & v > 0, ]
  pca <- prcomp(t(mat), scale. = TRUE, center = TRUE)
  pcs <- pca$x[, seq_len(min(10, ncol(pca$x))), drop = FALSE]
  d   <- dist(pcs)
  sb  <- mean(silhouette(as.integer(meta$study),     d)[, 3])
  sc  <- mean(silhouette(as.integer(meta$condition), d)[, 3])
  data.frame(
    method            = name,
    sil_by_study      = sb,
    sil_by_condition  = sc,
    score             = sc - sb
  )
})
metrics_df <- do.call(rbind, metrics)
metrics_df <- metrics_df[order(-metrics_df$score), ]
cat("\n=== Diagnostic metrics (sorted by score) ===\n")
print(metrics_df, row.names = FALSE)

write.csv(metrics_df, "data/diagnostics_metrics.csv", row.names = FALSE)
cat("\nSaved data/diagnostics_metrics.csv\n")