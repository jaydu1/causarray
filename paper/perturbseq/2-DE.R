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

path_base <- '~/../jinandmaya/perturbseq/'
setwd(path_base)



#'#######################################################################
# Load data
#'#######################################################################
sc.seurat.all <- readRDS("SCP1184/seurat-data.rds")
sc.seurat.all <- sc.seurat.all[,sc.seurat.all@meta.data$isAnalysed == TRUE]
celltype_filestr <- 'exneu'
sc.seurat.all <- subset(sc.seurat.all, subset = CellType == 'Excitatory')



path_rs <- sprintf('results/DE/')
dir.create(path_rs, recursive=TRUE, showWarnings = FALSE)

# Filter out rows with Perturbation count < 50
perturbation_counts <- table(sc.seurat.all@meta.data$Perturbation)
valid_perturbations <- names(perturbation_counts[perturbation_counts >= 50])
sc.seurat <- subset(sc.seurat.all, subset = Perturbation %in% valid_perturbations)

Y <- t(sc.seurat[['RNA']]$counts)
Y <- expm1(Y) / 1e6 * as.integer(sc.seurat@meta.data$nUMI)

sc.seurat <- subset(sc.seurat, features = Features(sc.seurat)[(rowSums(t(Y)>0)>50) & (rowMaxs(t(Y))>=10)])
sc.seurat <- FindVariableFeatures(sc.seurat, nfeatures=5000)

## Get metadata for individuals ----
metadata <- sc.seurat@meta.data
table(metadata$Perturbation)
#    Adnp    Ank2  Arid1b   Ash1l   Asxl3    Chd2    Chd8  Ctnnb1    Cul3   Ddx3x 
#     112      48      32     122     130      29      82     169     169      53 
#   Dscam  Dyrk1a  Fbxo11 Gatad2b     GFP   Kdm5b  Larp4b    Mbd5  Med13l    Mll1 
#      69      89     111      35     106      63      19     119      75     117 
#   Myst4    Pogz    Pten  Qrich1   Satb2  Scn2a1   Setd2   Setd5    Spen  Stard9 
#      84      56      93      86      51      93      76      71      63     151 
# Syngap1   Tcf20  Tcf7l2  Tnrc6b   Upf3b     Wac 
#      69      93      20     169     100      85 

Y <- Y[,VariableFeatures(sc.seurat)]
Y <- round(Y)

# Gene names to ensembl ----
gene_names <- toupper(colnames(Y))
mouse <- useMart("ensembl", dataset = "mmusculus_gene_ensembl", host='https://useast.ensembl.org')
gene_ensembls <- getBM(attributes = c('ensembl_gene_id', 'mgi_symbol'), 
  filters = 'mgi_symbol',
  values = gene_names,
  mart = mouse)
mapping <- setNames(gene_ensembls$ensembl_gene_id, gene_ensembls$mgi_symbol)

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



## Select covariates ----
# dummify the data
library(caret)
perturb <- metadata
colnames(perturb) <- gsub("Perturbation", "trt_", colnames(perturb))
perturb$trt_ <- relevel(as.factor(perturb$trt_), ref = "GFP")
dmy <- dummyVars(" ~ trt_", data = perturb)
perturb <- data.frame(predict(dmy, newdata = perturb))
dmy <- dummyVars(" ~ Batch", data = metadata)
batch <- data.frame(predict(dmy, newdata = metadata))

covs <- cbind(perturb[,-1], batch[,-dim(batch)[2]])




# Drop trt covariate...
covs.mx <- batch[,-dim(batch)[2]]
# Covariates need to be passed in as matrix... 
covs.mx <- matrix(as.numeric(unlist(covs.mx)),nrow=nrow(covs))
covs.mx <- matrix(nrow=nrow(Y), ncol=0)
A <- as.matrix(perturb[,-1])




#'#######################################################################
# DE analysis
#'#######################################################################


# DESeq ----
# Convert the factor levels to integers
metadata$Perturbation <- factor(metadata$Perturbation)
metadata$Perturbation <- relevel(metadata$Perturbation, ref = 'GFP')

metadata$trt <- factor(as.integer(metadata$Perturbation)  - 1)
correspondence <- data.frame(
  Integer_Level = (1:length(levels(metadata$Perturbation))) - 1,
  Original_Value = levels(metadata$Perturbation)
)

covs <- metadata['trt']

res.deseq <- run_DESeq(Y, covs, return_ds = TRUE)
df.deseq <- res.deseq$DESeq.df
sum(df.deseq$padj < 0.1, na.rm=T)

df.deseq$trt <- levels(metadata$Perturbation)[df.deseq$trt+1]
rownames(df.deseq) <- NULL
df.deseq$gene_names <- rep(colnames(Y), ifelse(length(dim(A)) == 2, dim(A)[2], 1))
df.deseq$ensembl_gene_id <- mapped_gene_names_to_ensembls(df.deseq$gene_names, mapping)

write.csv(df.deseq, paste0(path_rs, 'res.', celltype_filestr, '.deseq.csv'))
write.csv(rowData(res.deseq$DESeq.res), paste0(path_rs, 'stat.', celltype_filestr, '.deseq.csv'))




# RUV+DESeq ----
res.ruv <- run_ruv(Y, covs, 10)
df.ruv <- res.ruv[[1]]
df.ruv$trt <- levels(metadata$Perturbation)[df.ruv$trt+1]
sum(df.ruv$padj < 0.1, na.rm=T)

rownames(df.ruv) <- NULL
df.ruv$gene_names <- rep(colnames(Y), ifelse(length(dim(A)) == 2, dim(A)[2], 1))
df.ruv$ensembl_gene_id <- mapped_gene_names_to_ensembls(df.ruv$gene_names, mapping)

write.csv(df.ruv, paste0(path_rs, 'res.', celltype_filestr, '.ruv.csv'))
write.csv(rowData(res.ruv[[2]]$sce), paste0(path_rs, 'stat.', celltype_filestr, '.ruv.csv'))






# causarray ----
# # Select the number of unmeasured confounders
# res.causarray.r <- estimate_r_causarray(Y, covs.mx, A, seq(5,100,5))
# write.csv(res.causarray.r, paste0(path_rs, 'res.causarray.', celltype_filestr, '.r.csv'))
# fig <- causarray$plot_r(res.causarray.r[res.causarray.r$r<=100,])
# fig$savefig(paste0('res.causarray.', celltype_filestr, '.r.pdf'), dpi=300)


r <- 10
res.causarray <- run_causarray(Y, covs.mx, A, r=r, fdx=T, verbose=TRUE)

df.causarray <- res.causarray$causarray.df
df.causarray$trt <- levels(metadata$Perturbation)[df.causarray$trt+2]

sum(df.causarray$rej==1)
sum(df.causarray$padj < 0.1, na.rm=T)
sum(df.causarray$padj_emp_null_adj < 0.1, na.rm=T)

df.causarray$ensembl_gene_id <- mapped_gene_names_to_ensembls(df.causarray$gene_names, mapping)
write.csv(df.causarray, paste0(path_rs, 'res.', celltype_filestr, '.causarray.csv'))



path_rs <- sprintf('results/data/')
dir.create(path_rs, recursive=TRUE, showWarnings = FALSE)

data <- prep_causarray(Y, covs.mx, A)
Y_ <- data[[1]]; W <- data[[2]]; A <- data[[3]];

Wp <- res.causarray$causarray.res$W
write.csv(Y_, paste0(path_rs,  'Y.csv'))
write.csv(W, paste0(path_rs,  'W.csv'))
write.csv(Wp, paste0(path_rs,  'Wp.csv'))
write.csv(A, paste0(path_rs,  'A.csv'))