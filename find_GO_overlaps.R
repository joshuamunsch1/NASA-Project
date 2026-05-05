library(optparse)
option_list <- list(
  make_option(c("--file1"),
              type    = "character",
              default = "base_GO.csv",
              help    = "Path to first ORA result CSV [required]"),

  make_option(c("--file2"),
              type    = "character",
              default = "580.csv",
              help    = "Path to second ORA result CSV [required]"),

  make_option(c("-o", "--output"),
              type    = "character",
              default = "overlap_GO_terms.csv",
              help    = "Output CSV path for overlapping terms [default: overlap_GO_terms.csv]"),

  make_option(c("-s", "--sources"),
              type    = "character",
              default = NULL,
              help    = "Comma-separated GO sources to restrict analysis (e.g. GO:BP,GO:MF). Default: all sources.")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$file1) || is.null(opt$file2)) {
  stop("Both --file1 and --file2 must be specified. Run with --help for usage.")
}
for (f in c(opt$file1, opt$file2)) {
  if (!file.exists(f)) stop("File not found: ", f)
}

read_ora <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  required <- c("term_id", "term_name", "source", "p_value")
  missing  <- setdiff(required, colnames(df))
  if (length(missing) > 0) {
    stop("File '", path, "' is missing required columns: ", paste(missing, collapse = ", "))
  }
  df
}

label1 <- tools::file_path_sans_ext(basename(opt$file1))
label2 <- tools::file_path_sans_ext(basename(opt$file2))

df1 <- read_ora(opt$file1)
df2 <- read_ora(opt$file2)

df1 <- read.csv('pseudo_GO.csv', stringsAsFactors = FALSE, check.names = FALSE)
df2 <- read.csv('580_GO.csv', stringsAsFactors = FALSE, check.names = FALSE)
df3 <- read.csv('270_GO.csv', stringsAsFactors = FALSE, check.names = FALSE)
df4 <- read.csv('596_GO.csv', stringsAsFactors = FALSE, check.names = FALSE)

ids1 <- unique(df1$term_name)
ids2 <- unique(df2$term_name)
ids3 <- unique(df3$term_name)
ids4 <- unique(df4$term_name)

shared_ids2 <- intersect(ids1, ids2)
shared_ids3 <- intersect(ids1, ids3)
shared_ids4 <- intersect(ids1, ids4)

n1       <- length(ids1)
n2       <- length(ids2)
n3       <- length(ids3)
n4       <- length(ids4)
n_shared2 <- length(shared_ids2)
n_shared3 <- length(shared_ids3)
n_shared4 <- length(shared_ids4)

pct1 <- round(n_shared / n1 * 100, 1)   
pct2 <- round(n_shared2 / n2 * 100, 1)   
pct3 <- round(n_shared3 / n3 * 100, 1)
pct4 <- round(n_shared4 / n4 * 100, 1)

cat(sprintf("Fraction shared of 580: %.2f%%\n", 100 * n_shared2 / n2))
cat(sprintf("Fraction shared of 270: %.2f%%\n", 100 * n_shared3 / n3))
cat(sprintf("Fraction shared of 596: %.2f%%\n", 100 * n_shared4 / n4))

