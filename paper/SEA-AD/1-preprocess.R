library(Seurat)
library(reticulate)
library(anndata)
library(dplyr)
library(tidyverse)

set_meta_pb <- function(data, metadata, ind_name, cov_names){
    
    metadata$index <- metadata[,ind_name]
    
    metadata <- metadata[!duplicated(metadata$index),][,c('index',cov_names)]
    rownames(metadata) <- metadata$index
    metadata <- metadata[data@meta.data$orig.ident,]
    data@meta.data[,cov_names] <- metadata[,cov_names]
    data
}


args = commandArgs(trailingOnly=TRUE)
cat(args)
dataset <- as.character(args[1])

adata <- read_h5ad(sprintf("data/%s_exneu_sc_subsample.h5ad", dataset))
data <- CreateSeuratObject(counts = t(as.matrix(adata$X)), 
    meta.data = adata$obs, min.features = 0, min.cells = 0)
data@misc$var <- adata$var
rownames(data) <- adata$var$feature_name


Y <- t(as.matrix(data[['RNA']]$counts))
Y <- round(expm1(Y) / 1e4 * as.integer(data@meta.data$`Number of UMIs`))
data <- CreateSeuratObject(counts = t(Y), meta.data = data@meta.data, min.features = 0, min.cells = 0)

pb <- AggregateExpression(data, assays = "RNA", return.seurat = T, group.by = c("donor_id"))
pb <- set_meta_pb(pb, data@meta.data, "donor_id", c('Number of UMIs', 'disease', 'Cognitive status', 'sex', 'PMI', 'Age at death', 'self_reported_ethnicity'))
colnames(pb@meta.data) <- gsub(" ", "_", colnames(pb@meta.data))
pb <- subset(x=pb, subset = Cognitive_status != 'Reference')
pb[['RNA']]$data <- NULL
pb[['RNA']]$scale.data <- NULL


pb <- pb[rowSums(pb[['RNA']]$counts>0) >= 0,]
dim(pb)
# [1] 36412    80

pb <- pb[rowSums(pb[['RNA']]$counts>1) >= 40,]
dim(pb)
# [1] 29049    80


pb@meta.data$PMI <- sapply(pb@meta.data$PMI, function(x) {
  numbers <- as.numeric(str_extract_all(x, "\\d+\\.?\\d*")[[1]])
  mean(numbers)
})
pb@meta.data$Age_at_death <- sapply(pb@meta.data$Age_at_death, function(x) {
  numbers <- as.numeric(str_extract_all(x, "\\d+\\.?\\d*")[[1]])
  if (length(numbers) == 2) {
    mean(numbers)
  } else {
    90
  }
})

adata$var$feature_name <- gsub("_", "-", adata$var$feature_name)
pb@misc$var <- adata$var[adata$var$feature_name %in% rownames(pb),]

saveRDS(pb, sprintf("data/%s_exneu_pb.rds", dataset))
