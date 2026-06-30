# ============================================================
# 01_combat_ref.R  (ComBat-ref batch correction)
# ============================================================

suppressPackageStartupMessages({ library(edgeR) })

TISSUE       <- "liver"                                  # "liver" or "heart" 
PROJECT_ROOT   <- if (TISSUE == "liver") {
  "path/to/liver"
} else {
  "path/to/heart"
}

setwd(PROJECT_ROOT)

combat_ref_dir <- "path/to/Combat-ref"
src_main   <- file.path(combat_ref_dir, "ComBat_ref.R")
src_helper <- file.path(combat_ref_dir, "helper_seq.R")
if (!file.exists(src_main)) {
  stop("Cannot find ComBat_ref.R at ", src_main,
       "\nClone the repo first:  git clone https://github.com/xiaoyu12/Combat-ref")
}
source(src_main)
if (file.exists(src_helper)) source(src_helper)

out_dir <- "data/combat_ref"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# LOAD 
# ------------------------------------------------------------
counts <- readRDS("pipeline/merged_counts.rds")
meta   <- readRDS("pipeline/meta.rds")
stopifnot(identical(colnames(counts), as.character(meta$sample_id)))

batch     <- as.character(meta$study)
condition <- as.character(meta$condition)

corrected <- ComBat_ref(counts = counts, batch = batch, group = condition)

# ------------------------------------------------------------
# DIAGNOSTIC PCA 
# ------------------------------------------------------------
lc <- log2(corrected + 1); v <- apply(lc, 1, var)
pc <- prcomp(t(lc[v > 0, , drop = FALSE]), scale. = TRUE)
ve <- round(100 * (pc$sdev^2 / sum(pc$sdev^2))[1:2], 1)
png(file.path(out_dir, "combat_ref_pca.png"), width = 1500, height = 680, res = 150)
par(mfrow = c(1, 2))
plot(pc$x[,1], pc$x[,2], col = as.integer(meta$study), pch = 19,
     xlab = sprintf("PC1 (%.1f%%)", ve[1]), ylab = sprintf("PC2 (%.1f%%)", ve[2]),
     main = "ComBat-ref - by study")
legend("topright", levels(meta$study), col = seq_along(levels(meta$study)), pch = 19, cex = .7)
plot(pc$x[,1], pc$x[,2], col = c("navy","darkorange")[as.integer(meta$condition)], pch = 19,
     xlab = sprintf("PC1 (%.1f%%)", ve[1]), ylab = sprintf("PC2 (%.1f%%)", ve[2]),
     main = "ComBat-ref - by condition")
legend("topright", levels(meta$condition), col = c("navy","darkorange"), pch = 19, cex = .7)
dev.off()

write.csv(corrected, file.path(out_dir, "corrected_combat_ref.csv"))
