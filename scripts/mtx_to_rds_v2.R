suppressMessages({library(Matrix)})
D <- "data/refs/Oropharyngeal_HPVmix"
m <- readMM(file.path(D,"ref_counts_v2.mtx"))
rownames(m) <- readLines(file.path(D,"ref_genes_v2.txt"))
colnames(m) <- readLines(file.path(D,"ref_barcodes_v2.txt"))
m <- as(m,"CsparseMatrix"); storage.mode(m@x) <- "double"
saveRDS(m, file.path(D,"Oropharyngeal_UMI_v2.rds"))
cat(sprintf("rds: %d genes x %d cells\n", nrow(m), ncol(m)))
