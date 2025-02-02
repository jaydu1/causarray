library(Seurat)
library(SeuratObject)
library(DESeq2)
library(ggplot2)
library(scales)
library(qs)
library(dplyr)
library(zebrafishRNASeq)


#'#######################################################################
# Load method functions ----
#'#######################################################################
path_base <- '~/../jinandmaya/'
require(reticulate)
use_condaenv('renv')
setwd(paste0(path_base, 'methods'))
source('R_functions.R')

path_base <- '~/../jinandmaya/SEA-AD/'
celltype_filestr <- 'exneu'
setwd(path_base)

#'#######################################################################
# Load data
#'#######################################################################
# ps_model <- 'logistic' # 'random_forest_cv' # 

args = commandArgs(trailingOnly=TRUE)
cat(args)
ps_model <- as.character(args[1])
dataset <- as.character(args[2])
# ps_model <- 'logistic'

path_rs <- paste0(path_base, 'results-', dataset, '/', ps_model, '/DE/')
dir.create(path_rs, recursive=TRUE, showWarnings = FALSE)

pb.seurat_MTG <- readRDS(paste0(path_base, 'data/MTG_', celltype_filestr, '_pb.rds'))
genes <- rownames(pb.seurat_MTG)[rowMax(as.matrix(pb.seurat_MTG[['RNA']]$counts))>10]
pb.seurat_MTG <- subset(pb.seurat_MTG, features = genes)

pb.seurat_PFC <- readRDS(paste0(path_base, 'data/PFC_', celltype_filestr, '_pb.rds'))
genes <- rownames(pb.seurat_PFC)[rowMax(as.matrix(pb.seurat_PFC[['RNA']]$counts))>10]
pb.seurat_PFC <- subset(pb.seurat_PFC, features = genes)

genes <- intersect(rownames(pb.seurat_MTG), rownames(pb.seurat_PFC))
pb.seurat_MTG <- subset(pb.seurat_MTG, features = genes)
pb.seurat_PFC <- subset(pb.seurat_PFC, features = genes)

# pb.seurat <- merge(pb.seurat_MTG[genes], y = pb.seurat_PFC[genes], merge.data = TRUE, add.cell.ids = c("MTG", "PFC"))
if (dataset == 'MTG') {
    pb.seurat <- pb.seurat_MTG
} else {
    pb.seurat <- pb.seurat_PFC
}
pb.seurat <- subset(pb.seurat, subset = self_reported_ethnicity=='European')
pb.seurat@misc$var <- pb.seurat_PFC@misc$var
dim(pb.seurat)


genes <- toupper(genes)#rownames(pb.seurat)
rownames(pb.seurat) <- genes
# genes <- genes[rowMax(as.matrix(pb.seurat[['RNA']]$counts))>10]
# write.csv(data.frame(gene_names=genes), paste0(path_base, 'genes.csv'))

genes2 <- read.csv('../ROSMAP-AD/genes.csv')$gene_names
genes <- intersect(genes, genes2)

# Subset the Seurat object by genes
pb.seurat <- subset(pb.seurat, features = genes)

## Get metadata for individuals ----
pb.metadata <- pb.seurat@meta.data
pb <- pb.seurat[['RNA']]$counts
# Create column for treatment
pb.metadata$trt <- ifelse(pb.metadata$disease == 'normal', 0, 1)

## Select covariates ----
covs <- pb.metadata[, c('trt', 'PMI', 'Age_at_death', 'sex')]
# Make categorical variables factors
covs$sex <- factor(ifelse(covs$sex == 'male', 1, 0), levels=c(0, 1))
covs$PMI <- factor(covs$PMI, levels=c(4.55, 7.3, 10.05), labels=c(1, 2, 3))
covs$Age_at_death <- factor(covs$Age_at_death, levels=c(71, 83.5, 90), labels=c(1, 2, 3))
# covs <- covs %>% rename(CPS = `Continuous_Pseudo-progression_Score`)
# covs$self_reported_ethnicity <- factor(ifelse(covs$self_reported_ethnicity == 'European', 1, 0), levels=c(0, 1))
covs$trt <- factor(covs$trt, levels=c(0, 1))

covs.raw <- covs

# Scale and center quantitative variables
# covs$CPS <- scale(covs$CPS)
# covs <- covs[, c('trt', 'PMI', 'Age_at_death', 'sex', 'self_reported_ethnicity', 'tissue')]
covs <- covs[, c('trt', 'PMI', 'Age_at_death', 'sex')]


# Drop trt covariate...
# causarray.covs <- c('PMI', 'Age_at_death', 'sex', 'self_reported_ethnicity', 'tissue')
causarray.covs <- c('PMI', 'Age_at_death', 'sex')
covs.mx <- covs[, causarray.covs]
# Need to convert categorical variable (msex) back to 0, 1 (as.numeric(factor) converts incorrectly) - only two sexes here
covs.mx[, 'sex'] <- as.numeric(covs.mx[, 'sex'])-1
# covs.mx[, 'tissue'] <- as.numeric(covs.mx[, 'tissue'])-1
# covs.mx[, 'self_reported_ethnicity'] <- as.numeric(covs.mx[, 'self_reported_ethnicity'])-1
PMI_one_hot <- model.matrix(~ PMI - 1, data=covs)[,-1]
Age_at_death_one_hot <- model.matrix(~ Age_at_death - 1, data=covs)[,-1]
# covs.mx <- cbind(covs.mx[c('sex','self_reported_ethnicity','tissue')], PMI_one_hot, Age_at_death_one_hot)
covs.mx <- cbind(covs.mx[c('sex')], PMI_one_hot, Age_at_death_one_hot)

# Covariates need to be passed in as matrix... 
covs.mx <- matrix(as.numeric(unlist(covs.mx)), nrow=nrow(covs))


pb.seurat@misc$var <- pb.seurat@misc$var[match(colnames(t(pb)), toupper(pb.seurat@misc$var$feature_name)), ]

# Gene names to ensembl ----
gene_names <- colnames(t(pb))
gene_ids <- rownames(pb.seurat@misc$var)
# Create a named vector for mapping
mapping <- setNames(gene_ids, gene_names)
# Map gene_names to gene_ensembls, using an empty string for unmapped genes
mapped_gene_names_to_ensembls <- function(gene_names, mapping) {
  sapply(gene_names, function(gene) {
  if (gene %in% names(mapping)) {
    return(mapping[gene])
  } else {
    return("")
  }
})
}




#'#######################################################################
# DE analysis
#'#######################################################################

# DESeq ----
df.deseq <- run_DESeq(t(pb), covs)
sum(df.deseq$padj < 0.1, na.rm=T)
sum(is.na(df.deseq$padj))
df.deseq$gene_names <- rownames(df.deseq)
rownames(df.deseq) <- NULL
df.deseq$ensembl_gene_id <- mapped_gene_names_to_ensembls(df.deseq$gene_names, mapping)
write.csv(df.deseq, paste0(path_rs, 'res.', celltype_filestr, '.deseq.csv'))





# RUV+DESeq ----
ruv_r <- ruv::getK(as.matrix(t(pb)), as.matrix(as.numeric(covs$trt)-1), Z=covs.mx)
r <- min(ruv_r$k, 10)
cat('RUV k:', r, '\n')

df.ruv <- run_ruv(t(pb), covs, r)[[1]]
sum(df.ruv$padj < 0.1, na.rm=T)
df.ruv$gene_names <- rownames(df.ruv)
rownames(df.ruv) <- NULL
df.ruv$ensembl_gene_id <- mapped_gene_names_to_ensembls(df.ruv$gene_names, mapping)
write.csv(df.ruv, paste0(path_rs, 'res.', celltype_filestr, '.ruv.csv'))


# causarray ----

# Select the number of unmeasured confounders
# res.causarray.r <- estimate_r_causarray(t(pb), covs.mx, as.matrix(as.numeric(covs$trt)-1), seq(5,80,5))
# write.csv(res.causarray.r, paste0(path_rs, 'res.causarray.', celltype_filestr, '.r.csv'))
# fig <- causarray$plot_r(res.causarray.r[res.causarray.r$r<=100,])
# fig$savefig(paste0('res.causarray.', celltype_filestr, '.r.pdf'), dpi=300)

r <- 10
res.causarray <- run_causarray(t(pb), covs.mx, as.matrix(as.numeric(covs$trt)-1), r=r, verbose=TRUE,
    # glm_alpha=.1, shrinkage=T, 
    ps_model=ps_model, fdx=TRUE,
)

df.causarray <- res.causarray$causarray.df
sum(df.causarray$rej==1)
sum(df.causarray$padj < 0.1, na.rm=T)
sum(df.causarray$padj_emp_null_adj < 0.1, na.rm=T)

df.causarray$ensembl_gene_id <- mapped_gene_names_to_ensembls(df.causarray$gene_names, mapping)
write.csv(df.causarray, paste0(path_rs, 'res.', celltype_filestr, '.causarray.csv'))





path_rs <- paste0(path_rs, 'data/')
dir.create(path_rs, recursive=TRUE, showWarnings = FALSE)
data <- prep_causarray(t(pb), covs.mx, as.matrix(as.numeric(covs$trt)-1))
Y <- data[[1]]; A <- data[[3]]; W <- data[[2]]; 
Wp <- res.causarray$causarray.res$W
write.csv(Y, paste0(path_rs,  'Y.csv'))
write.csv(Wp, paste0(path_rs,  'Wp.csv'))
write.csv(W, paste0(path_rs,  'W.csv'))
write.csv(covs.raw, paste0(path_rs,  'W.raw.csv'))
write.csv(A, paste0(path_rs,  'A.csv'))
write.csv(res.causarray$causarray.res$pi, paste0(path_rs,  'pi.csv'))


