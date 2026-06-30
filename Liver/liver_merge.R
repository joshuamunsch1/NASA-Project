# ============================================================
# liver_merge.R
# ============================================================

setwd("path/to/liver")  

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
study_files <- c(
  "47"      = "47.csv",
  "168_rr1" = "168_RR1.csv",
  "168_rr3" = "168_RR3.csv",
  "242"     = "242.csv",
  "245"     = "245.csv",
  "379"     = "379.csv"
)

# Expected SF | GC per study from Table 1 (sanity check).
expected_counts <- list(
  "47"      = c(SF = 3,  GC = 3),
  "168_rr1" = c(SF = 5,  GC = 5),
  "168_rr3" = c(SF = 4,  GC = 4),
  "242"     = c(SF = 5,  GC = 13),
  "245"     = c(SF = 17, GC = 19),
  "379"     = c(SF = 27, GC = 32)
)

dir.create("pipeline", showWarnings = FALSE)

# ------------------------------------------------------------
# 1. LOAD + MERGE  (intersect on common ENSEMBL genes)
# ------------------------------------------------------------
read_study <- function(path) {
  m <- read.csv(path, row.names = 1, check.names = FALSE)
  colnames(m) <- gsub("\\.", "-", colnames(m))   # restore '-' in sample names
  as.matrix(m)
}

mats   <- lapply(study_files, read_study)
common <- Reduce(intersect, lapply(mats, rownames))

counts <- do.call(cbind, lapply(mats, function(m) m[common, , drop = FALSE]))
storage.mode(counts) <- "integer"

study <- factor(rep(names(study_files), vapply(mats, ncol, integer(1))),
                levels = names(study_files))

# ------------------------------------------------------------
# 2. CONDITION LABELS 
# ------------------------------------------------------------
condition <- factor(ifelse(grepl("FLT", colnames(counts)), "SF", "GC"),
                    levels = c("GC", "SF"))

meta <- data.frame(sample_id = colnames(counts),
                   study      = study,
                   condition  = condition,
                   row.names  = colnames(counts),
                   stringsAsFactors = TRUE)

cat("\nSF | GC counts by study (compare against Table 1):\n")
print(table(meta$study, meta$condition))

for (s in names(expected_counts)) {
  sub <- meta$condition[meta$study == s]
  got <- c(SF = sum(sub == "SF"), GC = sum(sub == "GC"))
  exp <- expected_counts[[s]]
  if (got["SF"] != exp["SF"] || got["GC"] != exp["GC"]) {
    warning(sprintf(
      "Study %s: got SF=%d GC=%d, expected SF=%d GC=%d -- check condition labels.",
      s, got["SF"], got["GC"], exp["SF"], exp["GC"]))
  }
}

# ------------------------------------------------------------
# 3. SAVE
# ------------------------------------------------------------
saveRDS(counts, "pipeline/merged_counts.rds")
saveRDS(meta,   "pipeline/meta.rds")