# ============================================================
# 03_deseq2.R 
# ============================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(limma)
})

TISSUE       <- "liver"                                  # "liver" or "heart" 
PROJECT_ROOT   <- if (TISSUE == "liver") {
  "path/to/liver"
} else {
  "path/to/heart"
}

out_dir <- "data/deseq2"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# LOAD 
# ------------------------------------------------------------
counts  <- readRDS("pipeline/merged_counts.rds")
meta    <- readRDS("pipeline/meta.rds")
stopifnot(identical(colnames(counts), as.character(meta$sample_id)))

coldata <- as.data.frame(meta)
rownames(coldata) <- coldata$sample_id

# ------------------------------------------------------------
# DESeqDataSet -> VST -> removeBatchEffect
# ------------------------------------------------------------
dds <- DESeqDataSetFromMatrix(countData = counts,
                              colData   = coldata,
                              design    = ~ study + condition)

vst_obj <- vst(dds, blind = FALSE)
vst_mat <- assay(vst_obj)

design    <- model.matrix(~ condition, data = meta)
corrected <- removeBatchEffect(x = vst_mat, batch = meta$study, design = design)

# ------------------------------------------------------------
# DIAGNOSTIC PCA 
# ------------------------------------------------------------
v  <- apply(corrected, 1, var)
pc <- prcomp(t(corrected[v > 0, , drop = FALSE]), scale. = TRUE)
ve <- round(100 * (pc$sdev^2 / sum(pc$sdev^2))[1:2], 1)
png(file.path(out_dir, "deseq2_pca.png"), width = 1500, height = 680, res = 150)
par(mfrow = c(1, 2))
plot(pc$x[,1], pc$x[,2], col = as.integer(meta$study), pch = 19,
     xlab = sprintf("PC1 (%.1f%%)", ve[1]), ylab = sprintf("PC2 (%.1f%%)", ve[2]),
     main = "DESeq2 VST+rbe - by study")
legend("topright", levels(meta$study), col = seq_along(levels(meta$study)), pch = 19, cex = .7)
plot(pc$x[,1], pc$x[,2], col = c("navy","darkorange")[as.integer(meta$condition)], pch = 19,
     xlab = sprintf("PC1 (%.1f%%)", ve[1]), ylab = sprintf("PC2 (%.1f%%)", ve[2]),
     main = "DESeq2 VST+rbe - by condition")
legend("topright", levels(meta$condition), col = c("navy","darkorange"), pch = 19, cex = .7)
dev.off()

write.csv(corrected, file.path(out_dir, "corrected_deseq2.csv"))
