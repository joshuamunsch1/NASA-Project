
counts_csv   <- "counts_complete.csv"           
meta_csv     <- "metadata.csv"
out_dir      <- "rna_integration_out"
gene_id_col  <- 1                             
condition_col <- "group"                    
dataset_col   <- "study"                    
alpha_fdr     <- 0.05                         
lfc_shrink_type <- "apeglm"                    
top_n_heatmap <- 50                           
set.seed(1)
setwd('C:/Users/Joshua/Downloads')

library(data.table)
library(ggplot2)
library(DESeq2)
library(edgeR)
library(pheatmap)
library(matrixStats)
library(fgsea)
library(EnhancedVolcano)
library(apeglm)
library(RColorBrewer)
library(biomaRt)
library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)


dt <- read.csv(counts_csv)
gene_ids <- dt[[gene_id_col]]
dt$X <- NULL
counts <- as.matrix(dt)
mode(counts) <- "numeric"

if (any(is.na(counts))) {
  warning("Counts contain NA; replacing with 0.")
  counts[is.na(counts)] <- 0
}
counts <- round(counts)

rownames(counts) <- make.unique(as.character(gene_ids))
colnames(counts) <- make.unique(colnames(counts))

meta <- read.csv(meta_csv)
meta <- as.data.frame(meta)

meta$sample <- as.character(meta$sample)
missing <- setdiff(colnames(counts), meta$sample)

meta <- meta[match(colnames(counts), meta$sample), , drop = FALSE]
cond_raw <- as.character(meta[[condition_col]])
u <- sort(unique(cond_raw))

meta[[condition_col]] <- factor(cond_raw, levels = c("0", "1"))
meta[[dataset_col]] <- factor(meta[[dataset_col]])

dge <- DGEList(counts = counts)
design_edger <- model.matrix(~ meta[[dataset_col]] + meta[[condition_col]])
keep <- filterByExpr(dge, design = design_edger)
message("Genes before: ", nrow(counts))
message("Genes after:  ", sum(keep))
counts_f <- counts[keep, , drop = FALSE]

dds <- DESeqDataSetFromMatrix(
  countData = counts_f,
  colData = meta,
  design = as.formula("~ group + study")
)
dds <- dds[rowSums(counts(dds)) > 0, ]

dds <- DESeq(dds, parallel = FALSE)

res <- results(dds, contrast = c("group", levels(meta$group)[2], levels(meta$group)[1]))

resLFC <- lfcShrink(dds, coef = resultsNames(dds)[which(grepl("group", resultsNames(dds)))][1],
                      type = "apeglm")

resOrdered <- resLFC[order(resLFC$padj), ]
write.csv(as.data.frame(resOrdered), file = "DE_results_full.csv")

# Summarize
summary(resOrdered)

resdf <- as.data.frame(resOrdered)
resdf$gene <- rownames(resdf)

EnhancedVolcano(resdf,
                  lab = resdf$gene,
                  selectLab = '',
                  x = 'log2FoldChange',
                  y = 'padj',
                  pCutoff = 0.05,
                  FCcutoff = 1,
                  title = "Volcano plot",
                  subtitle = paste0('Flight', " vs ", 'Ground'))
ggsave("volcano.png", width = 8, height = 6)

sig <- subset(resdf, padj < 0.05 & abs(log2FoldChange) >= 1)
write.csv(sig, file = "DE_significant_genes.csv", row.names = FALSE)


vsd <- vst(dds, blind = FALSE)  
vsd_mat <- assay(vsd)

pca <- prcomp(t(vsd_mat))
pca_df <- data.frame(
  sample_id = rownames(pca$x),
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  dataset = meta[[dataset_col]],
  condition = meta[[condition_col]]
)
p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = dataset, shape = condition)) +
  geom_point(size = 3) +
  theme_bw() +
  ggtitle("PCA on VST (blind=FALSE)")
ggsave(file.path(out_dir, "PCA_VST.png"), p_pca, width = 8, height = 6, dpi = 150)


topN <- 50
topgenes <- head(rownames(resOrdered), topN)
mat <- assay(vsd)[topgenes, ]
mat <- t(scale(t(mat)))  # row z-score
annotation_col <- as.data.frame(colData(vsd)[, setdiff(colnames(colData(vsd)), "replace") , drop = FALSE])

df_num <- data.frame(lapply(mat, function(col) {
  if (is.factor(col)) as.numeric(as.character(col)) else if (is.character(col)) as.numeric(col) else as.numeric(col)
}))

if (any(is.na(df_num))) {
  df_num[is.na(df_num)] <- 0
}

mat <- as.matrix(df_num)

pheatmap(mat, annotation_col = annotation_col, show_rownames = TRUE,
         filename = "heatmap_top50.png", main = paste("Top", topN, "DE genes"))


topN <- 20
topgenes <- head(rownames(resOrdered), topN)
mat <- assay(vsd)[topgenes, ]
mat <- t(scale(t(mat)))  # row z-score
annotation_col <- data.frame(
  group = factor(colData(vsd)[[condition_col]]),
  study = factor(colData(vsd)[[dataset_col]]),
  row.names = colnames(vsd)
)

pheatmap(mat, annotation_col = annotation_col, show_rownames = TRUE,
         filename = "heatmap_top50.png", main = paste("Top", topN, "DE genes"))


genes_input <- rownames(resOrdered)
ensembl = useMart("ensembl", dataset = "mmusculus_gene_ensembl")
annot <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol", "entrezgene_id", "description"),
               filters = "ensembl_gene_id", values = genes_input, mart = ensembl)

res_annot <- merge(as.data.frame(resOrdered), annot, by.x = "row.names", by.y = "ensembl_gene_id", all.x = TRUE)
colnames(res_annot)[1] <- "ensembl_id"
write.csv(res_annot, file = "DE_results_annotated.csv", row.names = FALSE)


sig_up <- subset(res_annot, padj < 0.05 & log2FoldChange >= 1 & !is.na(entrezgene_id))
sig_down <- subset(res_annot, padj < 0.05 & log2FoldChange <= -1 & !is.na(entrezgene_id))

if (nrow(sig_up) >= 10) {
  ego_up <- enrichGO(gene = unique(sig_up$entrezgene_id),
                     OrgDb = org.Hs.eg.db,
                     keyType = "ENTREZID",
                     ont = "BP",
                     pAdjustMethod = "BH",
                     pvalueCutoff = 0.05,
                     qvalueCutoff = 0.2)
  write.csv(as.data.frame(ego_up), "GO_BP_enrichment_up.csv", row.names = FALSE)
}
if (nrow(sig_down) >= 10) {
  ego_down <- enrichGO(gene = unique(sig_down$entrezgene_id),
                       OrgDb = org.Hs.eg.db,
                       keyType = "ENTREZID",
                       ont = "BP",
                       pAdjustMethod = "BH",
                       pvalueCutoff = 0.05,
                       qvalueCutoff = 0.2)
  write.csv(as.data.frame(ego_down), "GO_BP_enrichment_down.csv", row.names = FALSE)
}

if (nrow(sig_up) >= 10) {
  ekegg_up <- enrichKEGG(gene = unique(sig_up$entrezgene_id),
                         organism = "hsa", pvalueCutoff = 0.05)
  write.csv(as.data.frame(ekegg_up), "KEGG_enrichment_up.csv", row.names = FALSE)
}


if (!is.null(res_annot$entrezgene_id)) {
  rank_df <- res_annot[!is.na(res_annot$entrezgene_id), ]
  ranks <- rank_df$log2FoldChange
  names(ranks) <- as.character(rank_df$entrezgene_id)
  ranks <- sort(ranks, decreasing = TRUE)

  go2gene <- as.list(org.Hs.egGO2ALLEGS)
  fgseaRes <- fgsea::fgsea(pathways = go2gene, stats = ranks, nperm = 1000)
  fgseaRes <- fgseaRes[order(padj), ]
  write.csv(as.data.frame(fgseaRes), file = "GSEA_results.csv", row.names = FALSE)
}


