########################################################################
#
# Load the Python package
#
########################################################################

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
run_causarray <- function(Y, W, A, r, update_disp_glm=TRUE, W_A='full', ...){
    cat(causarray$'__version__','\n')
    
    data <- prep_causarray(Y, W, A, ...)
    Y <- data[[1]]; W <- data[[2]]; A <- data[[3]]; offset <- data[[4]];
    d <- dim(W)[2]; a <- ifelse(length(dim(A)) == 2, dim(A)[2], 1)

    # confounder adjustment
    if(r>0){
        X <- cbind(W, A)
        res_gcate <- causarray$fit_gcate(Y, X, r, num_d=a, offset=offset, ...)
        # res is a list of 2 matrices:
        # A1 = [X, Z] and A2 = [B, Gamma] where X = [W, A]
        names(res_gcate) <- c('A1', 'A2', 'info', 'A01', 'A02')
        Wp <- res_gcate[[1]]
        Wp <- Wp[,-c((d+1):(d+a))] # remove the column of A
        disp_glm <- res_gcate[[3]]$kwargs_glm$nuisance
        Z <- Wp[,(d+1):ncol(Wp)]
        offset <- log(res_gcate[[3]]$kwargs_glm$size_factor)
    }else{
        res_gcate <- NULL
        Wp <- W
        disp_glm <- NULL
        Z <- NULL
    }
        
    if(update_disp_glm){
        disp_glm <- NULL
    }

    W_new <- cbind(W, Wp[,-c(1:d)])
    if(W_A=='full'){        
        W_A <- cbind(W, Wp[,-c(1:d)], data[[5]])
    }else if(is.null(W_A)){
        W_A <- W_new
    }else{
        W_A <- matrix(1, nrow=nrow(Y), ncol=1)
    }
    res <- do.call(
      causarray$LFC,
      modifyList(list(alpha=0.1, c=0.1, family='nb', offset=offset, disp_glm=disp_glm), 
          list('Y'=Y, 'W'=W_new, 'A'=A, 'W_A'=W_A, ...))
      )
    causarray.df <- res[[1]]
    
    causarray.df$gene_names <- rep(colnames(Y), ifelse(length(dim(A)) == 2, dim(A)[2], 1))
    
    causarray.res <- c(res_gcate, res[[2]])
    causarray.res$Z <- Z
    causarray.res$W <- Wp
    causarray.res$Y <- Y

    return(list('causarray.df'=causarray.df, 'causarray.res'=causarray.res))    
}

estimate_r_causarray <- function(Y, W, A, r_max, ...){
    data <- prep_causarray(Y, W, A, ...)
    Y <- data[[1]]; W <- data[[2]]; A <- data[[3]]
    df_r <- causarray$estimate_r(Y, cbind(W, A), r_max, ...)
    df_r
}



prep_causarray <- function(Y, W, A, intercept=TRUE, offset=TRUE, ...){
    Y <- as.matrix(Y)
    if (is.null(W)) {
      W <- matrix(0, nrow = nrow(Y), ncol = 0)
    } else {
      W <- as.matrix(W)
    }
    A <- as.matrix(A)
    Y <- pmin(Y, round(quantile(apply(Y, MARGIN=c(2), max), 0.999)))

    if(intercept){
        intercept <- rep(1, nrow(W))
    }else{
        intercept <- NULL
    }    

    W <- cbind(intercept, W)
    W <- as.matrix(W)

    # size_factor <- causarray$comp_size_factor(Y, ...)
    loglibsize <- scale(log2(rowSums(Y)))

    return(list(Y, W, A, offset, loglibsize))
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
library(MASS)

.check_input <- function(Y, metadata, indvs=NULL){
  n_indvs <- length(unique(indvs))
  n_meta <- dim(metadata)[1]
  if (((dim(Y)[1] != n_meta) || (n_meta != n_meta)) && (length(indvs) != dim(Y)[1])) {
    stop('Dimensions do not match for Y and metadata. Is Y individual x gene?')
  } 
  if (is.null(colnames(metadata))) {
    stop('metadata must have column names.')
  }
  if (!('trt' %in% colnames(metadata))) {
    stop('Treatment assignment column `trt` not in metadata.')
  }

}

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
run_wilcoxon <- function(Y, metadata, raw=F, family='nb') {
  .check_input(Y, metadata)

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
  
  .check_input(Y, metadata)

  t0 <- Sys.time()
  dds <- DESeqDataSetFromMatrix(countData = t(Y), 
                                colData = metadata,
                                design = reformulate(colnames(metadata)))
  # dds <- DESeq(dds)
  # to avoid NA p-values
  geoMeans <- exp(MatrixGenerics::rowMeans(log(counts(dds))))
  if(all(geoMeans==0)){
      geoMeans <- apply(counts(dds), 1, function(row) if (all(row == 0)) 0 else exp(mean(log(row[row != 0]))))
  }
  sizeFactors(dds) <- estimateSizeFactorsForMatrix(counts(dds), geoMeans=geoMeans)
  dds <- DESeq(dds, minReplicatesForReplace=Inf)
  # equivalently
  # dds <- estimateSizeFactors(dds, geoMeans=geoMeans)
  # dds <- estimateDispersions(dds)
  # dds <- nbinomWaldTest(dds)
  t1 <- Sys.time()
  print(t1-t0)
  
  # res.DESeq2 <- results(dds, name="trt_1_vs_0")
  # to avoid NA q-values
  if (nlevels(metadata$trt) > 2) {
      res.DESeq2 <- data.frame()
      for (i in 1:(nlevels(metadata$trt)-1)) {
          res <- results(dds, name=sprintf("trt_%d_vs_0", i), cooksCutoff=cooksCutoff, independentFiltering=independentFiltering)
          res <- as.data.frame(res)
          res$trt <- i
          res.DESeq2 <- rbind(res.DESeq2, res)
      }
  } else {
      res.DESeq2 <- results(dds, name="trt_1_vs_0", cooksCutoff=cooksCutoff, independentFiltering=independentFiltering)
      res.DESeq2 <- as.data.frame(res.DESeq2)
  }

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
  for (gene in rownames(res.cocoa$resid.ln.mu)) {
    wilc <- wilcox.test(res.cocoa$resid.ln.mu[gene, ] ~ res.cocoa$indv2exp$exp)
    
    cocoa_stats <- c(cocoa_stats, wilc$statistic)
    cocoa_pvals <- c(cocoa_pvals, wilc$p.value)    
  }
  cocoa_qvals <- qvalue(cocoa_pvals)$qvalues
  df.cocoa <- data.frame('stat' = cocoa_stats,
                        'pvalue' = cocoa_pvals,
                        'padj' = cocoa_qvals)
  
  rownames(df.cocoa) <- rownames(res.cocoa$resid.ln.mu)
  
  return(list('cocoA.df'=df.cocoa,
              'cocoA.res'=res.cocoa))
}


#' Function that prepares and runs CocoA-Diff on single-cell data (prep_cocoa and fit_cocoa)
#' 
#' @param sc Single-cell gene expression matrix, cell x gene
#' @param metadata Metadata dataframe (indv x cov) containing trt column where 1 denotes treatment/case and 0 denotes control
#' @param indvs Individual names for each cell, default NULL
#' @param cocoAWriteName File path denoting where to save files
#' @returns List of CocoA-Diff results, `cocoA.df` dataframe contains Wilcoxon statistic, p-value, and adjusted p-value
#'
run_cocoa <- function(sc, metadata, indvs = NULL, cocoAWriteName='simu/tmp'){

    .check_input(sc, metadata, indvs)

    A <- metadata$trt
    if(is.null(indvs)){
        indvs <- 1:dim(sc)[1]
    }

    cell2indv <- data.frame(cell=1:dim(sc)[1], indv=indvs)

    indv2trt <- data.frame(indv=indvs[!duplicated(indvs)],
                        exp=A)

    # prep_cocoa needs to take in a S4 matrix
    seuratY <- CreateSeuratObject(counts=sc, project = "simu", assay = "RNA",
                    min.cells = 0, min.features = 0, names.field = 1,
                    names.delim = "_", meta.data = NULL)
    
    if(packageVersion("Seurat")>"5"){
        s4Y <- seuratY[['RNA']]@layers$counts # Seurat v5
    }else{
        s4Y <- seuratY@assays$RNA@counts # Seurat v4
    }
    
    colnames(s4Y) <- 1:dim(sc)[2]#paste0('gene', 1:dim(sc)[2])
    rownames(s4Y) <- 1:dim(sc)[1]#paste0('cell', 1:dim(sc)[1])
    mtx.data <- prep_cocoa(s4Y, cocoAWriteName)

    res.cocoa <- fit_cocoa(mtx.data, cell2indv, indv2trt)
    return(res.cocoa)
}


library(RUVSeq)
library(zebrafishRNASeq)

# RUVr ----
#' Function that runs RUVr on single-cell data
#' 
#' @param Y Gene expression matrix,individual x gene
#' @param metadata Metadata dataframe with all covariates including column called trt with 1 = treatment/case and 0 = control (factor),
#'                 numeric features should be centered and scaled, all covariates in this dataframe will be used
#' @param r The number of unmeasured confounders to be estimated.
#' @returns Dataframe containing results from DESeq2 including test statistics, p-values, log2 fold changes, etc.
#'
run_ruv <- function(Y, metadata, r, ...) {
    cat('Running RUV... \n')
    .check_input(Y, metadata)

    pb <- t(Y)
    if(is.null(rownames(pb))){
        rownames(pb) <- 1:dim(pb)[1]
    }
    genes <- rownames(pb)
    gpath <- metadata$trt
    set <- newSeqExpressionSet(as.matrix(pb),
                                phenoData = data.frame(gpath, row.names=colnames(pb)))
    design <- model.matrix(~gpath, data=pData(set))

    y <- DGEList(counts=counts(set))
    y <- calcNormFactors(y, method="TMM")
    y <- estimateGLMCommonDisp(y, design)
    y <- estimateGLMTagwiseDisp(y, design)
    fit <- glmFit(y, design)
    res <- residuals(fit, type="deviance")

    set4 <- RUVr(set, genes, k=r, res)
    w <- pData(set4)

    pb2 <- SingleCellExperiment(assays=list(counts=pb), colData=metadata)
    for(j in 1:length(w)){
      pb2[[paste0('w_', j)]] <- w[[paste0('W_', j)]]
    }
    
    # DE analysis
    formula <- ~trt + .
    cd <- as.data.frame(colData(pb2))
    design <- model.matrix(formula, cd)

    res.deseq <- run_DESeq(t(pb), cd, return_ds=TRUE, ...)
    ruv.df <- res.deseq[[1]]
    ruv.res <- list()
    dds <- res.deseq[[2]]
    
    x <- DESeq2:::getModelMatrix(dds)[,-2]
    beta <- as.matrix(coef(dds))[,-2]

    ruv.res$Z <- w[,colnames(w)!='gpath']
    ruv.res$W <- colData(dds)[,colnames(colData(dds))!='trt']
    ruv.res$Y_hat_0 <- 2^(x %*% t(beta)) * sizeFactors(dds)
    ruv.res$sce <- dds
    return(list('ruv.df'=ruv.df, 'ruv.res'=ruv.res))
}




library(ruvIIInb)
library(SingleCellExperiment)
library(DelayedArray)

# RUV-III-NB ----
#' Function that runs RUV-III-NB on single-cell data
#' 
#' @param Y Single-cell gene expression matrix,individual x gene
#' @param metadata Metadata dataframe with all covariates including column called trt with 1 = treatment/case and 0 = control (factor),
#'                 numeric features should be centered and scaled, all covariates in this dataframe will be used
#' @param ctl Vector of gene names to be used as controls
#' @param r The number of unmeasured confounders to be estimated.
#' @returns Dataframe containing results from DESeq2 including test statistics, p-values, log2 fold changes, etc.
#'
run_ruv3nb <- function(Y, metadata, ctl, r, batch=NULL) {
    cat('Running RUV3-NB... \n')
    
    .check_input(Y, metadata)

    t0 <- Sys.time()
    sce <- SingleCellExperiment(assays=list(counts=t(Y)), colData=metadata)
    rowData(sce)$ctlLogical <- ctl

    # Perform initial clustering to identify pseudo-replicates
    sce <- scran::computeSumFactors(sce,assay.type="counts")
    data_norm_pre <- sweep(assays(sce)$counts,2,sce$sizeFactor,'/')
    assays(sce, withDimnames=FALSE)$lognormcounts<- log(data_norm_pre+1)
    snn_gr_init <- scran::buildSNNGraph(sce, assay.type = "lognormcounts")
    clusters_init <- igraph::cluster_louvain(snn_gr_init)
    sce$cluster_init <- factor(clusters_init$membership)

    # Construct the replicate matrix M using pseudo-replicates identified using initial clustering
    M <- matrix(0,ncol(assays(sce)$counts),length(unique(sce$cluster_init)))
    cl <- sort(unique(as.numeric(unique(sce$cluster_init))))
    for(CL in cl){M[which(as.numeric(sce$cluster_init)==CL),CL] <- 1}

    #RUV-III-NB code
    ruv3nb_out <- tryCatch(
        {
        ruv3nb_out <- fastruvIII.nb(
            Y=DelayedArray(assays(sce)$counts), # count matrix with genes as rows and cells as columns
            M=M, #Replicate matrix constructed as above
            ctl=rowData(sce)$ctlLogical, #A vector denoting control genes
            k=r,
            use.pseudosample=TRUE,
            batch=batch,
            ncores = 6
        )
        },
        error=function(e) {
          ruv3nb_out <- fastruvIII.nb(
            Y=DelayedArray(assays(sce)$counts), # count matrix with genes as rows and cells as columns
            M=M, #Replicate matrix constructed as above
            ctl=rowData(sce)$ctlLogical, #A vector denoting control genes
            k=r,
            use.pseudosample=FALSE,
            batch=batch,
            ncores = 6
          )
        }
    )

    sce_ruv3nb <- makeSCE(ruv3nb_out, cData=colData(sce))

    markers_ruv3nb <- scran::findMarkers(
      x=as.matrix(assays(sce_ruv3nb)$logPAC),
      groups=sce_ruv3nb$trt, test.type="wilcox"
      )

    res.ruv3nb <- markers_ruv3nb[[2]]
    res.ruv3nb <- res.ruv3nb[,-ncol(res.ruv3nb)]
    rownames(res.ruv3nb) <- as.integer(rownames(res.ruv3nb))
    res.ruv3nb <- res.ruv3nb[sort(rownames(res.ruv3nb)), ]
    colnames(res.ruv3nb) <- c('top', 'pvalue', 'padj', 'stat')

    t1 <- Sys.time()
    print(t1-t0)

    Z <- ruv3nb_out$W
    W <- cbind(ruv3nb_out$M, ruv3nb_out$W)
    
    return(list('ruv3nb.df'=res.ruv3nb,
                'ruv3nb.res'=list('Z'=Z, 'W'=W, 'sce'=sce_ruv3nb)))
  
}