library(Seurat)
library(SeuratObject)
library(DESeq2)
library(ggplot2)
library(scales)
library(qs)
library(dplyr)
library(zebrafishRNASeq)
library("biomaRt")


#'#######################################################################
# Load method functions ----
#'#######################################################################
path_base <- '~/../jinandmaya/'
require(reticulate)
use_condaenv('renv')
setwd(paste0(path_base, 'methods'))
source('R_functions.R')

path_base <- '~/../jinandmaya/ROSMAP-AD/'
celltype_filestr <- 'exneu'
setwd(path_base)

#'#######################################################################
# Load data
#'#######################################################################
# ps_model <- 'logistic' # 'random_forest_cv' # 

args = commandArgs(trailingOnly=TRUE)
cat(args)
ps_model <- as.character(args[1])

path_rs <- paste0(path_base, 'results/', ps_model, '/DE/')
dir.create(path_rs, recursive=TRUE, showWarnings = FALSE)


pb.seurat <- readRDS(paste0(path_base, 'data/', celltype_filestr, '_pb.rds'))
rownames(pb.seurat) <- toupper(rownames(pb.seurat))
genes <- rownames(pb.seurat)
genes <- genes[rowMax(as.matrix(pb.seurat[['RNA']]$counts))>10]
# write.csv(data.frame(gene_names=genes), paste0(path_base, 'genes.csv'))

genes2 <- read.csv('../SEA-AD/genes.csv')$gene_names
genes <- intersect(genes, genes2)

# Subset the Seurat object by genes
pb.seurat <- subset(pb.seurat, features = genes)
dim(pb.seurat)

pb.seurat <- pb.seurat[,is.na(pb.seurat@meta.data$pmi) | (pb.seurat@meta.data$pmi<50)]
dim(pb.seurat)


## Get metadata for individuals ----
pb.metadata <- pb.seurat@meta.data
pb <- pb.seurat[['RNA']]$counts
# Create column for treatment
pb.metadata$trt <- ifelse(pb.metadata$age_first_ad_dx == '', 0, 1)


## Select covariates ----
covs <- pb.metadata[, c('trt', 'msex', 'pmi', 'age_death')]
# Make categorical variables factors
covs$msex <- factor(ifelse(covs$msex == 1, 1, 0), levels=c(0, 1))
covs$trt <- factor(covs$trt, levels=c(0, 1))

# Remove NAs
covs$pmi[is.na(covs$pmi)] <- median(covs$pmi[!is.na(covs$pmi)])

# Turn quantitative covariates into floats (rather than strings)
covs$age_death[covs$age_death == '90+'] <- '90'
covs$age_death <- as.numeric(covs$age_death)
covs.raw <- covs

# Scale and center quantitative variables
covs[c('pmi', 'age_death')] <- lapply(covs[c('pmi', 'age_death')], function(x) c(scale(x)))
covs <- covs[, c('trt', 'msex', 'pmi', 'age_death')]


# Drop trt covariate...
causarray.covs <- c('msex', 'pmi', 'age_death')
covs.mx <- covs[, causarray.covs]
# Need to convert categorical variable (msex) back to 0, 1 (as.numeric(factor) converts incorrectly) - only two sexes here
covs.mx[, 'msex'] <- as.numeric(covs.mx[, 'msex'])-1
# Covariates need to be passed in as matrix... 
covs.mx <- matrix(as.numeric(unlist(covs.mx)),nrow=nrow(covs))


# Gene names to ensembl ----
gene_names <- toupper(colnames(t(pb)))
hsmart <- useMart(dataset = "hsapiens_gene_ensembl", biomart = "ensembl", host='https://useast.ensembl.org')

gene_ensembls <- getBM(attributes = c('ensembl_gene_id', 'hgnc_symbol'), 
  filters = 'hgnc_symbol',
  values = gene_names,
  mart = hsmart)

# Create a named vector for mapping
mapping <- setNames(gene_ensembls$ensembl_gene_id, gene_ensembls$hgnc_symbol)

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
res.causarray.r <- estimate_r_causarray(t(pb), covs.mx, as.matrix(as.numeric(covs$trt)-1), seq(5,100,5))
write.csv(res.causarray.r, paste0(path_rs, 'res.causarray.', celltype_filestr, '.r.csv'))
fig <- causarray$plot_r(res.causarray.r[res.causarray.r$r<=100,])
fig$savefig(paste0('res.causarray.', celltype_filestr, '.r.pdf'), dpi=300)


r <- 10
res.causarray <- run_causarray(t(pb), covs.mx, as.matrix(as.numeric(covs$trt)-1), r=r, verbose=TRUE,
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