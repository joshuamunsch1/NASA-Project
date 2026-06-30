# ============================================================
# 02_ruvg.R  (RUVg batch correction)
# ============================================================

suppressPackageStartupMessages({
  library(RUVSeq)
  library(edgeR)
})

TISSUE       <- "liver"                                  # "liver" or "heart" 
PROJECT_ROOT   <- if (TISSUE == "liver") {
  "path/to/liver"
} else {
  "path/to/heart"
}
setwd(PROJECT_ROOT)

k        <- 4       
n_top_de <- 5000    
out_dir  <- "data/ruvg"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
counts <- readRDS("pipeline/merged_counts.rds")
meta   <- readRDS("pipeline/meta.rds")
stopifnot(identical(colnames(counts), as.character(meta$sample_id)))

# ------------------------------------------------------------
# EMPIRICAL NEGATIVE CONTROL GENES
# Genes with the LEAST evidence of condition DE (after adjusting
# for study) serve as in-silico controls.
# ------------------------------------------------------------
dge    <- DGEList(counts = counts, group = meta$condition)
dge    <- calcNormFactors(dge, method = "TMM")
design <- model.matrix(~ study + condition, data = meta)
dge    <- estimateDisp(dge, design)
fit    <- glmQLFit(dge, design)

cond_coef <- grep("^condition", colnames(design))   # the SF term, NOT coef=1
qlf       <- glmQLFTest(fit, coef = cond_coef)

top    <- topTags(qlf, n = Inf)$table
n_excl <- min(n_top_de, nrow(top))
control_genes <- rownames(counts)[!rownames(counts) %in% rownames(top)[seq_len(n_excl)]]


# ------------------------------------------------------------
# RUN RUVg
# ------------------------------------------------------------
ruv_out   <- RUVg(x = counts, cIdx = control_genes, k = k, isLog = FALSE, round = TRUE)
corrected <- ruv_out$normalizedCounts
W         <- ruv_out$W

# ------------------------------------------------------------
# DIAGNOSTIC PCA 
# ------------------------------------------------------------
lc <- log2(corrected + 1); v <- apply(lc, 1, var)
pc <- prcomp(t(lc[v > 0, , drop = FALSE]), scale. = TRUE)
ve <- round(100 * (pc$sdev^2 / sum(pc$sdev^2))[1:2], 1)
png(file.path(out_dir, sprintf("ruvg_pca_k%d.png", k)), width = 1500, height = 680, res = 150)
par(mfrow = c(1, 2))
plot(pc$x[,1], pc$x[,2], col = as.integer(meta$study), pch = 19,
     xlab = sprintf("PC1 (%.1f%%)", ve[1]), ylab = sprintf("PC2 (%.1f%%)", ve[2]),
     main = sprintf("RUVg k=%d - by study", k))
legend("topright", levels(meta$study), col = seq_along(levels(meta$study)), pch = 19, cex = .7)
plot(pc$x[,1], pc$x[,2], col = c("navy","darkorange")[as.integer(meta$condition)], pch = 19,
     xlab = sprintf("PC1 (%.1f%%)", ve[1]), ylab = sprintf("PC2 (%.1f%%)", ve[2]),
     main = "RUVg - by condition")
legend("topright", levels(meta$condition), col = c("navy","darkorange"), pch = 19, cex = .7)
dev.off()

write.csv(corrected, file.path(out_dir, "corrected_ruvg_counts.csv"))
saveRDS(W,           file.path(out_dir, "ruvg_factors.rds"))
