########################################################################
#
# Load the Python package
#
########################################################################
# require(reticulate)

# # create a new environment 
# virtualenv_create("py")
# virtualenv_install("py", c("numba", "pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "cvxpy", "tqdm"))
# use_virtualenv("py")
# See the following document for more details
# https://rstudio.github.io/reticulate/articles/versions.html


# import Python package
causarray <- import("causarray")

# causarray ----
#' Function that runs causarray test on pseudo-bulk data
#' 
#' @param Y Pseudo-bulk gene expression matrix, individual x gene
#' @param W Scaled covariate matrix (without intercept or loglibsize), individual x n_cov
#' @param A Treatment, individual x 1
#' @param r The number of unmeasured confounder, 0 if no unmeasured confounder
#' @param alpha First of two parameters to control P(FDP > c) <= alpha.
#' @param c Second of two parameters to control P(FDP > c) <= alpha.
#' @returns Dataframe containing Wilcoxon statistic and p-value for each gene
#'
run_causarray <- function(Y, W, A, r, alpha=0.05, c=0.1, func='LFC', family='poisson', ...){
    data <- prep_causarray(Y, W, A)
    Y <- data[[1]]; W <- data[[2]]; A <- data[[3]]
    d <- dim(W)[2]

    # confounder adjustment
    if(r>0){
        res <- causarray$fit_gcate(Y, cbind(W, A), r, family=family, ...)
        # res is a list of 2 matrices:
        # A1 = [X, Z] and A2 = [B, Gamma] where X = [W, A]
        Wp <- res[[1]]        
        Wp <- Wp[,-(d+1)] # remove the column of A
        disp_glm <- res[[3]]$kwargs_glm$nuisance
        Z <- Wp[,(d+1):ncol(Wp)]
    }else{
        Wp <- W
        disp_glm <- NULL
        Z <- NULL
    }
    
    # different estimands: log-fold change, fold change, average treatment effects, and standarized average treatment effects
    if(func=='LFC'){
        res <- causarray$LFC(Y, Wp, A, alpha=alpha, c=c, family=family, disp_glm=disp_glm, ...)
    }else if(func=='FC'){
        res <- causarray$FC(Y, Wp, A, alpha=alpha, c=c, family=family, disp_glm=disp_glm, ...)
    }else if(func=='ATE'){
        res <- causarray$ATE(Y, Wp, A, alpha=alpha, c=c, family=family, disp_glm=disp_glm, ...)
    }else if(func=='SATE'){
        res <- causarray$SATE(Y, Wp, A, alpha=alpha, c=c, family=family, disp_glm=disp_glm, ...)
    }

    causarray.df <- res[[1]]
    causarray.res <- res[[2]]
    causarray.res$Z <- Z

    return(list('causarray.df'=causarray.df, 'causarray.res'=causarray.res))
}

estimate_r_causarray <- function(Y, W, A, r_max, family='poisson', ...){
    data <- prep_causarray(Y, W, A)
    Y <- data[[1]]; W <- data[[2]]; A <- data[[3]]
    df_r <- causarray$estimate_r(Y, cbind(W, A), r_max, family=family, ...)
    df_r
}



prep_causarray <- function(Y, W, A, ...){
    Y <- as.matrix(Y)
    W <- as.matrix(W)
    A <- as.matrix(A)
    loglibsize <- scale(log2(rowSums(Y)))
    intercept <- rep(1, nrow(W))
    W <- cbind(intercept, W, loglibsize)
    W <- as.matrix(W)

    return(list(Y, W, A))
}


outcome_model <- function(dds){
  # get design matrix
  modelMatrix <- DESeq2:::getModelMatrix(dds)


  # get column names for design matrix
  strsplit(names(mcols(dds))[startsWith(names(mcols(dds)), 'WaldStatistic_')], "_")
  x <- names(mcols(dds))[startsWith(names(mcols(dds)), 'WaldStatistic_')]
  x <- regmatches(x, regexpr("_", x), invert = TRUE)
  modelMatrixname <- sapply(x, "[[", 2)

  # get estimated coefficients
  hbeta <- mcols(dds)[,modelMatrixname]
  hbeta <- as.matrix(hbeta)

  normalizationFactors <- DESeq2:::getSizeOrNormFactors(dds)
  # mu <- normalizationFactors * t(exp(modelMatrix %*% t(hbeta) / log2(exp(1))))


  X1 <- cbind(modelMatrix[,-ncol(modelMatrix)], matrix(1, nrow = nrow(modelMatrix)))
  X0 <- cbind(modelMatrix[,-ncol(modelMatrix)], matrix(0, nrow = nrow(modelMatrix)))

  mu0 <- as.matrix(t(normalizationFactors * t(exp(X0 %*% t(hbeta) / log2(exp(1))))))
  mu1 <- as.matrix(t(normalizationFactors * t(exp(X1 %*% t(hbeta) / log2(exp(1))))))

  return(list(mu0,mu1))
}



# It requires extra pacakge "scanpy"
source_python('cinemaot.py')
# run_cinemaot returns list(df, TE)



require(qvalue)
library(DESeq2)
library(mmutilR)
library(Seurat)
library(SeuratObject)
require(doParallel) # for parallel processing
library(data.table)
library(stringr)

# Wilcoxon ----
#' Function that runs Wilcoxon test on pseudo-bulk data (residuals from poisson regression)
#' 
#' @param Y Pseudo-bulk gene expression matrix, individual x gene
#' @param metadata Metadata dataframe with all covariates including column called trt with 1 = treatment/case and 0 = control (factor),
#'                 numeric features should be centered and scaled, all covariates in this dataframe will be used
#' @param raw Whether the input data is raw or preprocessed
#' @param family What time of regression to fit? 'poisson' or 'nb' (negative binomial)
#' @returns Dataframe containing Wilcoxon statistic, p-value, and adjusted p-value for each gene
#'
run_wilcoxon <- function(Y, metadata, raw=F, family='poisson') {

  if (dim(Y)[1] != dim(metadata)[1]) {
    stop('Dimensions do not match for Y and metadata. Is Y individual x gene?')
  } 
  if (is.null(colnames(metadata))) {
    stop('metadata must have column names.')
  }
  if (!('trt' %in% colnames(metadata))) {
    stop('Treatment assignment column `trt` not in metadata.')
  }

  # if (raw) {
  #   Y <- sweep(Y,1,rowSums(Y),`/`)
  #   Y <- log2((Y*10000)+1)
  # } 
  loglibsize <- log2(rowSums(Y))
  intercept <- rep(1, nrow(Y))
  metadata <- cbind(intercept, metadata, loglibsize)
  
  if (is.null(colnames(Y))) {
    cycle <- 1:ncol(Y)
  } else {
    cycle <- colnames(Y)
  }
  
  cat(paste0('Fitting ', str_to_title(family), ' Regressions... \n'))
  t0 <- Sys.time()
  fit_regs <- function(gene, Y, metadata, family='poisson') {
    gene_data <- cbind(Y[,gene], metadata)
    colnames(gene_data) <- c('gexp', colnames(metadata))
    if(family=='poisson'){
      reg <- glm(gexp ~ . - trt, family="poisson", data=gene_data)
    }else if(family=='nb'){
      reg <- glm.nb(gexp ~ . - trt, data=gene_data)
    }    
    reg$residuals
  }  
  Y_resid <- do.call("cbind", mclapply(cycle,fit_regs,Y=Y,metadata=metadata,family=family,mc.cores=20,mc.preschedule = T,mc.silent=T))
  colnames(Y_resid) <- cycle

  cat('Running Wilcoxon Tests... \n')
  wilcox_test <- function(gene, Y, trt){
    wilc <- wilcox.test(Y[trt == 1, gene], Y[trt == 0, gene])
    list('stat' = wilc$statistic, 'pvalue' = wilc$p.value)
  }
  test <- mclapply(cycle,wilcox_test,Y=Y_resid,trt=metadata$trt,mc.cores=20,mc.preschedule = T,mc.silent=T)
  wilc_df <- rbindlist(test)
  wilc_df <- as.data.frame(wilc_df)
  wilc_df['padj'] <- qvalue(wilc_df['pvalue'])$qvalues
  rownames(wilc_df) <- cycle
  t1 <- Sys.time()
  print(t1-t0)
  
  return(wilc_df)
}

# DESeq ----
#' Function that runs DESeq on single-cell data
#' 
#' @param Y Single-cell gene expression matrix,individual x gene
#' @param metadata Metadata dataframe with all covariates including column called trt with 1 = treatment/case and 0 = control (factor),
#'                 numeric features should be centered and scaled, all covariates in this dataframe will be used
#' @returns Dataframe containing results from DESeq2 including test statistics, p-values, log2 fold changes, etc.
#'
run_DESeq <- function(Y, metadata, cooksCutoff=FALSE, independentFiltering=FALSE, return_ds=FALSE) {
  cat('Running DESeq... \n')

  if (dim(Y)[1] != dim(metadata)[1]) {
    stop('Dimensions do not match for Y and metadata. Is Y individual x gene?')
  }
  if (is.null(colnames(metadata))) {
    stop('metadata must have column names.')
  }
  if (!('trt' %in% colnames(metadata))) {
    stop('Treatment assignment column `trt` not in metadata.')
  }
  
  t0 <- Sys.time()
  dds <- DESeqDataSetFromMatrix(countData = t(Y), 
                                colData = metadata,
                                design = reformulate(colnames(metadata)))
  # dds <- DESeq(dds)
  # to avoid NA p-values
  dds <- DESeq(dds, minReplicatesForReplace=Inf)
  t1 <- Sys.time()
  print(t1-t0)
  
  # res.DESeq2 <- results(dds, name="trt_1_vs_0")
  # to avoid NA q-values
  res.DESeq2 <- results(dds, name="trt_1_vs_0", cooksCutoff=cooksCutoff, independentFiltering=independentFiltering)
  res.DESeq2 <- as.data.frame(res.DESeq2)
  if(return_ds == FALSE){
    return(res.DESeq2)
  }else{
    return(list('DESeq.df'=res.DESeq2,
              'DESeq.res'=dds))
  }
  
}

# CocoA-Diff ----
#' Function that prepares files to run CocoA-Diff
#' 
#' @param sc Single-cell gene expression matrix, cell x gene
#' @param cocoAWriteName File path denoting where to save files
#' @returns mtx.data
#'
prep_cocoa <- function(sc, cocoAWriteName) {
  
  dir.create(dirname(cocoAWriteName), recursive=TRUE, showWarnings = FALSE)

  sc <- t(sc)
  
  # Remove existing files
  fileNames <- c('.mtx.gz', '.cols.gz', '.rows.gz', '.mtx.gz.index')
  for (fileName in fileNames) {
    if (file.exists(paste0(cocoAWriteName, fileName))) {
      file.remove(paste0(cocoAWriteName, fileName))
    }
  }
  
  # Write new files
  mtx.data <- write.sparse(sc, rownames(sc), colnames(sc), cocoAWriteName) 
  
  return(mtx.data)
}

#' Function that runs CocoA-Diff on single-cell data
#' 
#' @param mtx.data Output from prep_cocoa
#' @param cell2indv Dataframe mapping from cell to individual
#' @param indv2trt Dataframe mapping from individual to treatment
#' @returns List of CocoA-Diff results, `cocoA.df` dataframe contains Wilcoxon statistic, p-value, and adjusted p-value
#'
fit_cocoa <- function(mtx.data, cell2indv, indv2trt) {
  
  t0 <- Sys.time()
  res.cocoa <- make.cocoa(mtx.data=mtx.data, 
                          celltype="bulk", 
                          cell2indv=cell2indv, 
                          indv2exp=indv2trt, 
                          knn = 50)
  t1 <- Sys.time()
  print(t1-t0)
  
  cocoa_stats <- c()
  cocoa_pvals <- c()
  for (gene in rownames(res.cocoa$resid.mu)) {
    wilc <- wilcox.test(res.cocoa$resid.mu[gene, ] ~ res.cocoa$indv2exp$exp)
    
    cocoa_stats <- c(cocoa_stats, wilc$statistic)
    cocoa_pvals <- c(cocoa_pvals, wilc$p.value)    
  }
  cocoa_qvals <- qvalue(cocoa_pvals)$qvalues
  df.cocoa <- data.frame('stat' = cocoa_stats,
                        'pvalue' = cocoa_pvals,
                        'padj' = cocoa_qvals)
  
  rownames(df.cocoa) <- rownames(res.cocoa$resid.mu)
  
  return(list('cocoA.df'=df.cocoa,
              'cocoA.res'=res.cocoa))
}


#' Function that prepares and runs CocoA-Diff on single-cell data (prep_cocoa and fit_cocoa)
#' 
#' @param sc Single-cell gene expression matrix, cell x gene
#' @param metadata Metadata dataframe containing trt column where 1 denotes treatment/case and 0 denotes control
#' @param indvs Individual names for each cell, default NULL
#' @param cocoAWriteName File path denoting where to save files
#' @returns List of CocoA-Diff results, `cocoA.df` dataframe contains Wilcoxon statistic, p-value, and adjusted p-value
#'
run_cocoa <- function(
    sc, metadata, indvs = NULL, cocoAWriteName='simu/tmp'){

    A <- metadata$trt
    if(is.null(indvs)){
        indvs <- 1:dim(sc)[1]
    }
    cell2indv <- data.frame(cell=1:dim(sc)[1], indv=indvs)
    indv2trt <- data.frame(indv=indvs[!duplicated(indvs)], exp=A)

    # prep_cocoa needs to take in a S4 matrix
    seuratY <- CreateSeuratObject(counts=sc, project = "simu", assay = "RNA",
                    min.cells = 0, min.features = 0, names.field = 1,
                    names.delim = "_", meta.data = NULL)
    
    if(packageVersion("Seurat")>"5"){
        s4Y <- seuratY[['RNA']]@layers$counts # Seurat v5
    }else{
        s4Y <- seuratY@assays$RNA@counts # Seurat v4
    }
    
    colnames(s4Y) <- 1:dim(sc)[2]
    rownames(s4Y) <- 1:dim(sc)[1]
    mtx.data <- prep_cocoa(s4Y, cocoAWriteName)

    res.cocoa <- fit_cocoa(mtx.data, cell2indv, indv2trt)
    return(res.cocoa)
}



