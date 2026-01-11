# Dataset C: complex heterogeneity
set.seed(2026)
G <- 2000
studies <- c("A","B","C")
samples_per_study <- c(10,10,10)
N <- sum(samples_per_study)

# metadata
coldata <- data.frame(sample = paste0("c",1:N),
                      study = rep(studies, samples_per_study),
                      stringsAsFactors = FALSE)
# random group balanced
coldata$group <- sample(c("ctrl","case"), size=N, replace=TRUE)

# baseline: mixture of gammas so some very-highly expressed genes
base_mean <- c(rgamma(G*0.9, shape=1.5, scale=30), rgamma(G*0.1, shape=5, scale=200))[1:G]

# library sizes per study (different)
study_sf <- c(A=0.8, B=1.0, C=1.3)
lib_sf <- sapply(coldata$study, function(s) rlnorm(1, meanlog=log(study_sf[s]), sdlog=0.25))

# create gene modules (co-regulated)
n_modules <- 6
module_genes <- split(sample(1:G), rep(1:n_modules, length.out=G))
# module effects per study
module_effects <- matrix(1, nrow=G, ncol=length(studies), dimnames=list(NULL,studies))
for (m in seq_along(module_genes)) {
  genes_m <- module_genes[[m]]
  # each study perturbs some modules differently:
  module_effects[genes_m, "A"] <- runif(length(genes_m), 0.7, 1.4)
  module_effects[genes_m, "B"] <- runif(length(genes_m), 0.8, 1.6)
  module_effects[genes_m, "C"] <- runif(length(genes_m), 0.6, 2.0)
}

# dispersion
phi <- 0.03 + 0.12 * runif(G)  # mix of low-high

# DE genes by group (12%), some module-specific DE
is_de <- sample(c(rep(TRUE, floor(0.12*G)), rep(FALSE, G - floor(0.12*G))))
fold_changes <- ifelse(is_de, runif(G, 1.4, 4.0), 1.0)

# build mu
mu_mat <- matrix(0, nrow=G, ncol=N)
for (j in 1:N) {
  s <- coldata$study[j]
  mu0 <- base_mean * lib_sf[j] * module_effects[, s]
  if (coldata$group[j] == "case") mu0 <- mu0 * fold_changes
  mu_mat[, j] <- mu0
}

# sample counts
counts <- matrix(0, nrow=G, ncol=N)
for (i in 1:G) {
  size <- 1 / phi[i]
  counts[i, ] <- rnbinom(N, size=size, mu = mu_mat[i, ])
}

# inject dropouts: randomly set some low counts to zero with probability depending on mu
low_idx <- which(mu_mat < 5, arr.ind = TRUE)
sel <- sample(seq_len(nrow(low_idx)), size = floor(0.02 * nrow(low_idx)))
for (k in sel) counts[ low_idx[k,1], low_idx[k,2] ] <- 0

# outlier samples: pick 1–2 samples and inflate counts
outliers <- sample(1:N, 2)
counts[, outliers] <- counts[, outliers] * sample(3:8, length(outliers), replace=TRUE)

rownames(counts) <- paste0("gene", 1:G)
colnames(counts) <- coldata$sample

# Add spike-in genes with known scale (simulate 10 spike genes)
spike_genes <- paste0("spike",1:10)
spike_counts <- matrix(rpois(10*N, lambda=500), nrow=10, ncol=N)
rownames(spike_counts) <- spike_genes
colnames(spike_counts) <- coldata$sample

# final combined matrix
counts_full <- rbind(counts, spike_counts)

write.csv(
  counts_full,
  file = "dataset_C_counts.csv",
  quote = FALSE
)
write.csv(
  coldata,
  file = "dataset_C_coldata.csv",
  row.names = FALSE,
  quote = FALSE
)
