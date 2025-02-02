suppressPackageStartupMessages({
    library(splatter)
    library(scater)
})
library(stats)
library(rhdf5)
library(tidyr)
library(stringr)

# n_list <- c(100, 200)
# args <- commandArgs(trailingOnly = TRUE)
# n <- 200 #n_list[1+as.numeric(args[1])]

args = commandArgs(trailingOnly=TRUE)
if(length(args)==0){
    ind <- ''
}else{
    ind <- args[1]
}

if((ind != '') && grepl( 'd', ind, fixed = TRUE)){
    d <- as.integer(str_extract(ind, "(?<=d_)[0-9]*"))
}else{
    d <- 0
}

if((ind != '') && grepl( 'r', ind, fixed = TRUE)){
    r <- as.integer(str_extract(ind, "(?<=r_)[0-9]*"))
}else{
    r <- 0
}

if((ind != '') && grepl( 'noise', ind, fixed = TRUE)){
    if(grepl( '.', ind, fixed = TRUE)){
        noise <- as.numeric(str_extract(ind, "(?<=noise_)\\d+\\.?\\d+"))
    }else{
        noise <- as.numeric(str_extract(ind, "(?<=noise_)\\d+"))
    }
    
}else{
    noise <- 1
}

path_base = '/home/jinandmaya/simu_nb/'

for(n in seq(100, 300, 100)){
    path_data <- sprintf(paste0(path_base,'data/simu_%d%s/'), n, ind)
    dir.create(path_data, recursive=TRUE, showWarnings = FALSE)
    for(seed in c(0:49)){
        cat(n, seed, '\n')
        set.seed(seed)
        
        n_batch <- as.integer((d+r + 1)/2)
        batchCells <- rep(floor(n/n_batch), n_batch)
        batchCells[1] <- batchCells[1] + n - sum(batchCells)

        # Simulate data using estimated parameters
        sim <- splatSimulate(
                seed=seed, 
                
                # treatment effect
                group.prob = c(0.5, 0.5), method = "groups", verbose=F,
                de.prob=0.05, de.facLoc=1., de.facScale=0.5, de.downProb=0.5,

                # dropout
                dropout.type="experiment", dropout.mid=20, dropout.shape=0.001,
                
                # batch effect
                batchCells = batchCells, batch.facLoc=noise, batch.facScale=0.5,
                
                mean.shape=0.3, mean.rate=0.6,
                nGenes=10000, lib.loc=11, lib.scale=.2,
                bcv.common = 0.1, bcv.df = 60,
                )

        theta <- (rowData(sim)$DEFacGroup1 !=1) | (rowData(sim)$DEFacGroup2 !=1)
        Y <- as.matrix(t(counts(sim)))
        theta <- theta[colSums(Y>10)>10]
        Y <- Y[,colSums(Y>10)>10]
        A <- 1 * (colData(sim)$Group == 'Group1')
        W <- model.matrix(~0+colData(sim)$Batch)[,-n_batch]
        W <- cbind(rep(1,n), W)
        colnames(W) <- c('intercept', paste0("batch_", 1:(n_batch-1)))
        W <- data.frame(W)
        cat(dim(Y), sum(theta), sum(A), dim(W), '\n')

        metadata <- data.frame(
            'cell'=1:n,
            'indv'=1:n,
            'trt'=A,
            'celltype'=readr::parse_number(colData(sim)$Batch)
        )

        h5write(t(Y), sprintf('%ssimu_data_%d.h5', path_data, seed), '/Y')
        h5write(t(W), sprintf('%ssimu_data_%d.h5', path_data, seed), '/W')
        h5write(A, sprintf('%ssimu_data_%d.h5', path_data, seed), '/A')
        h5write(theta, sprintf('%ssimu_data_%d.h5', path_data, seed), '/theta')
        h5write(t(metadata), sprintf('%ssimu_data_%d.h5', path_data, seed), '/metadata')
    }
}