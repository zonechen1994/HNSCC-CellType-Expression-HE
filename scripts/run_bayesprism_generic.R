#!/usr/bin/env Rscript
# 通用 BayesPrism 反卷积——换癌种只改 JSON 配置，R 侧不用动。
# 用法: Rscript run_bayesprism_generic.R <config.json> <n_samples|all> <n_cores>
#
# config.json 字段(见 notebook 配置区自动生成):
#   ref_dir          单细胞参考所在目录
#   annotation_file  细胞注释表(.txt/.txt.gz/.csv): 含 barcode、大类、(可选)亚型、(可选)来源 列
#   umi_rds          原始 UMI 计数矩阵 .rds (genes x cells, 列名=barcode)
#   barcode_col      注释表里的 barcode 列名 (要与 umi 矩阵列名对应)
#   celltype_col     大类细胞列名 (cell.type)
#   subtype_col      亚型列名 (可空; use_subtype_as_state=TRUE 时用作 cell.state)
#   origin_col       样本来源列名 (可空; 用于只保留肿瘤来源细胞)
#   origin_keep      要保留的来源值数组 (可空)
#   drop_types       要剔除的大类数组 (可空)
#   malignant_key    恶性细胞的大类名 (BayesPrism 的 key 参数)
#   bulk_file        TCGA bulk 计数矩阵 .tsv.gz (第一列=基因, 其余列=样本)
#   out_dir          输出目录
#   use_subtype_as_state  TRUE=用"大类_亚型"复合标签当 cell.state(细亚型优化版)
suppressMessages({library(BayesPrism); library(Matrix); library(data.table); library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
cfg <- fromJSON(args[1])
n_samples <- ifelse(length(args)>=2, args[2], "all")
n_cores   <- ifelse(length(args)>=3, as.integer(args[3]), 16L)
dir.create(cfg$out_dir, showWarnings=FALSE, recursive=TRUE)

cat("== load annotation ==\n")
ann <- fread(file.path(cfg$ref_dir, cfg$annotation_file))
if (!is.null(cfg$origin_col) && length(cfg$origin_keep)>0)
  ann <- ann[get(cfg$origin_col) %in% cfg$origin_keep]
if (length(cfg$drop_types)>0)
  ann <- ann[!get(cfg$celltype_col) %in% cfg$drop_types]
cat(sprintf("  reference cells: %d; types: %s\n", nrow(ann),
            paste(sort(unique(ann[[cfg$celltype_col]])), collapse=", ")))
keep_bc <- ann[[cfg$barcode_col]]

cat("== load raw UMI (.rds, genes x cells) & subset ==\n")
m <- readRDS(file.path(cfg$ref_dir, cfg$umi_rds))
sc <- m[, colnames(m) %in% keep_bc]; rm(m); gc()
sc <- as.matrix(Matrix::t(sc))                       # cells x genes
idx <- match(rownames(sc), ann[[cfg$barcode_col]])
ct  <- ann[[cfg$celltype_col]][idx]
storage.mode(sc) <- "double"
cat(sprintf("  sc: %d cells x %d genes\n", nrow(sc), ncol(sc)))

# cell.state: 默认=大类; 细亚型版=大类_亚型 复合(保证 state 唯一归属一个 type)
if (isTRUE(cfg$use_subtype_as_state) && !is.null(cfg$subtype_col)) {
  st <- paste(ct, ann[[cfg$subtype_col]][idx], sep="_")
} else st <- ct

cat("== load bulk ==\n")
bk <- fread(cfg$bulk_file)
bg <- bk[[1]]; bk <- as.matrix(bk[, -1, with=FALSE]); rownames(bk) <- bg; bk <- t(bk)
if (n_samples != "all") bk <- bk[1:min(as.integer(n_samples),nrow(bk)), , drop=FALSE]
storage.mode(bk) <- "double"
cat(sprintf("  bulk: %d samples x %d genes\n", nrow(bk), ncol(bk)))

cat("== cleanup + prism ==\n")
sc.f <- cleanup.genes(input=sc, input.type="count.matrix", species="hs",
                      gene.group=c("Rb","Mrp","other_Rb","chrM","MALAT1"), exp.cells=5)
sc.f <- as.matrix(sc.f); storage.mode(sc.f) <- "double"
myPrism <- new.prism(reference=sc.f, mixture=bk, input.type="count.matrix",
                     cell.type.labels=ct, cell.state.labels=st,
                     key=cfg$malignant_key, outlier.cut=0.01, outlier.fraction=0.1)
cat(sprintf("== run.prism n.cores=%d ==\n", n_cores))
bp <- run.prism(prism=myPrism, n.cores=n_cores)
saveRDS(bp, file.path(cfg$out_dir,"bp_res.rds"))
theta <- get.fraction(bp=bp, which.theta="final", state.or.type="type")
write.csv(theta, file.path(cfg$out_dir,"theta.csv"))
for (cell in colnames(theta)) {
  Z <- get.exp(bp=bp, state.or.type="type", cell.name=cell)
  write.csv(Z, file.path(cfg$out_dir, sprintf("Z_%s.csv", gsub("[^A-Za-z0-9]+","_",cell))))
  cat(sprintf("  %s: %d x %d\n", cell, nrow(Z), ncol(Z)))
}
cat("ALL_DONE\n")
