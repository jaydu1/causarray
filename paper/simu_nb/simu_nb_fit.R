library(rhdf5)
# Load Python and R functions for basic Wilcoxon Tests, DESeq2, and CocoA-Diff
# and causarray, cinemaot
require(reticulate)
use_condaenv('renv')
path_base = '/home/jinandmaya/'
setwd(paste0(path_base, 'methods'))
source('R_functions.R')
path_base = '/home/jinandmaya/simu_nb/'
setwd(path_base)
library(stringr)

n <- 200

c = 0.1
alpha = 0.1

ind <- '_d_1_r_4_noise_0.1'
num_r <- 1

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
    num_r <- as.integer(str_extract(ind, "(?<=r_)[0-9]*"))
}else{
    num_r <- 0
}



for(n in seq(100, 300, 100)){
    path_result <- sprintf(paste0(path_base,'results/simu_%d%s/'), n, ind)
    dir.create(path_result, recursive=TRUE, showWarnings = FALSE)

    for (seed in 0:49) {
        path_data <- sprintf(paste0(path_base,'data/simu_%d%s/simu_data_%d.h5'), n, ind, seed)

        if(file.exists(sprintf('%scausarray_r_%d_%d.csv', path_result, 6, seed))){
            next
        }

        Y <- t(h5read(path_data, '/Y'))
        metadata <- t(h5read(path_data, '/metadata'))
        W <- t(h5read(path_data, '/W'))  
        if(num_r>0){
            W <- W[, 1:(d+1), drop = FALSE]
        }
        A <- h5read(path_data, '/A')
        theta <- h5read(path_data, '/theta')
        
        # Dropping first column because all 1s -> all NAs when scaling
        if(ncol(W)>1){
            scaleW <- as.data.frame(scale(W))
            scaleW <- scaleW[, 2:ncol(scaleW), drop = FALSE]
            colnames(scaleW) <- paste0('X', 1:ncol(scaleW))
        }else{
            scaleW <- data.frame(trt=A)
        }
        scaleW$trt <- factor(A, levels=c(0, 1))
        
        possibleError <- tryCatch(
            {
        # Wilcoxon Tests ----
        res.wilc <- run_wilcoxon(Y, metadata=scaleW, raw=T)

        # DESeq ----
        res.DESeq <- run_DESeq(Y, metadata=scaleW, return_ds=TRUE)
        dds <- res.DESeq[[2]]
        res.DESeq <- res.DESeq[[1]]

        # CocoA-Diff ----
        cocoa <- run_cocoa(sc = Y, indvs=metadata[,2], metadata=scaleW, cocoAWriteName=sprintf('%stmp_cocoa/tmp_%d', path_result, seed))
        res.cocoa <- cocoa[[1]]
        cf.cocoa <- t(cocoa[[2]]$cf.ln.mu)
        t(cocoa[[2]]$resid.ln.mu)
        

        # CINEMA-OT ---- run_cinemaot returns list(df, CF, TE)
        res.cinemaot <- run_cinemaot(Y, A, raw=TRUE)
        cf.cinemaot <- res.cinemaot[[2]]$Y_hat_0
        W.cinemaot <- res.cinemaot[[2]]$W
        res.cinemaot <- res.cinemaot[[1]]

        res.cinemaotw <- run_cinemaot(Y, A, raw=TRUE, weighted=TRUE)
        cf.cinemaotw <- res.cinemaotw[[2]]$Y_hat_0
        W.cinemaotw <- res.cinemaotw[[2]]$W
        res.cinemaotw <- res.cinemaotw[[1]]

        # Save confounder estimation results
        write.csv(cf.cocoa, sprintf('%scocoa_cf_%d.csv', path_result, seed))        
        write.csv(cf.cinemaot, sprintf('%scinemaot_cf_%d.csv', path_result, seed))
        write.csv(W.cinemaot, sprintf('%scinemaot_W_%d.csv', path_result, seed))
        write.csv(cf.cinemaotw, sprintf('%scinemaotw_cf_%d.csv', path_result, seed))
        write.csv(W.cinemaotw, sprintf('%scinemaotw_W_%d.csv', path_result, seed))

        # Save test results
        write.csv(res.wilc, sprintf('%swilc_%d.csv', path_result, seed))
        write.csv(res.DESeq, sprintf('%sDESeq_%d.csv', path_result, seed))
        write.csv(res.cocoa, sprintf('%scocoa_%d.csv', path_result, seed))
        write.csv(res.cinemaot, sprintf('%scinemaot_%d.csv', path_result, seed))
        write.csv(res.cinemaotw, sprintf('%scinemaotw_%d.csv', path_result, seed))

        for(r_hat in c(2,4,6)){
            # RUV
            ruv <- run_ruv(Y, metadata=scaleW, r_hat)
            res.ruv <- ruv[[1]]
            cf.ruv <- log1p(ruv[[2]]$Y_hat_0)
            W.ruv <- ruv[[2]]$W

            # RUV-3-NB
            ctl <- rep(FALSE, ncol(Y))
            ctl[sort(sample(which(theta==0), (ncol(Y)/3), replace=FALSE))] <- TRUE
            ruv3nb <- run_ruv3nb(Y, metadata=scaleW, ctl, r_hat)
            res.ruv3nb <- ruv3nb[[1]]
            cf.ruv3nb <- t(assays(ruv3nb[[2]]$sce)$logPAC)
            W.ruv3nb <- ruv3nb[[2]]$W

            # causarray
            # res.causarray.r <- estimate_r_causarray(Y, scaleW[,-ncol(scaleW)], A, 5)
            # r_hat <- res.causarray.r[which.min(res.causarray.r$JIC), 'r']
            # cat('r_hat:', r_hat, '\n')
            # r_hat <- num_r
            # res.causarray <- run_causarray(Y, scaleW[,-ncol(scaleW)], A, 
            #     alpha=alpha, c=c, r=r_hat, #B=2000,
            #     # Y_hat_0=res[[1]], Y_hat_1=res[[2]],
            #     # ps_model='logistic', penalty=NULL,
            #     # max_depth=as.integer(5), func='LFC'
            #     )[[1]]

            res.causarray <- run_causarray(Y, scaleW[,-ncol(scaleW)], A, 
                fdx=T, r=r_hat, glm_alpha=.5, shrinkage=T)
            cf.causarray <- log1p(res.causarray[[2]]$Y_hat_0)
            W.causarray <- res.causarray[[2]]$W
            res.causarray <- res.causarray[[1]]

            # Save confounder estimation results
            write.csv(cf.ruv, sprintf('%sruv_r_%d_cf_%d.csv', path_result, r_hat, seed))
            write.csv(W.ruv, sprintf('%sruv_r_%d_W_%d.csv', path_result, r_hat, seed))
            write.csv(cf.ruv3nb, sprintf('%sruv3nb_r_%d_cf_%d.csv', path_result, r_hat, seed))
            write.csv(W.ruv3nb, sprintf('%sruv3nb_r_%d_W_%d.csv', path_result, r_hat, seed))
            write.csv(cf.causarray, sprintf('%scausarray_r_%d_cf_%d.csv', path_result, r_hat, seed))
            write.csv(W.causarray, sprintf('%scausarray_r_%d_W_%d.csv', path_result, r_hat, seed))

            write.csv(res.ruv, sprintf('%sruv_r_%d_%d.csv', path_result, r_hat, seed))
            write.csv(res.ruv3nb, sprintf('%sruv3nb_r_%d_%d.csv', path_result, r_hat, seed))
            write.csv(res.causarray, sprintf('%scausarray_r_%d_%d.csv', path_result, r_hat, seed))
        }


        },
        error=function(e) e
        )
    }
}