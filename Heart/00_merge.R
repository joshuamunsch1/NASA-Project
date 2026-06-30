# ============================================================
# 00_merge.R
# ============================================================

setwd("path/to/heart")  
# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
study_files <- c(
  "270"     = "270.csv",
  "580"     = "580.csv",
  "599"     = "599.csv"
)

# Expected SF | GC per study from Table 1 (sanity check).
expected_counts <- list(
  
  "270"     = c(SF = 3,  GC = 2),   # GLDS-270: bulk heart, F1/F2/F9 vs G7/G8
  "580"     = c(SF = 20, GC = 20),  # GLDS-573: Basal/Vivarium arms dropped upstream
  "599"     = c(SF = 3,  GC = 3)    # GLDS-596
)

dir.create("pipeline", showWarnings = FALSE)

# ------------------------------------------------------------
# 1. LOAD + MERGE  
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
condition <- ifelse(grepl("FLT", colnames(counts)), "SF", "GC")

# OSD-599 override: pull the authoritative SF set from 599.txt.
md599_path <- file.path("data", "single_study_metadata", "599.txt")
if (file.exists(md599_path)) {
  md599 <- read.delim(md599_path, header = TRUE, sep = "\t")
  sf599 <- as.character(
    md599$Sample.Name[grepl("Space", md599$Factor.Value.Spaceflight.)])
  sf599 <- gsub("\\.", "-", gsub("-", ".", sf599))  # match merged colname form
  sel <- study == "599"
  condition[sel] <- ifelse(colnames(counts)[sel] %in% sf599, "SF", "GC")
} else {
  warning("599.txt not found; 599 labels fall back to the FLT-token rule.")
}

condition <- factor(condition, levels = c("GC", "SF"))

meta <- data.frame(sample_id = colnames(counts),
                   study      = study,
                   condition  = condition,
                   row.names  = colnames(counts),
                   stringsAsFactors = TRUE)


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
