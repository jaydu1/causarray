# install.packages("BiocManager")
# BiocManager::install("clusterProfiler")
# BiocManager::install("org.Hs.eg.db")
# BiocManager::install("AnnotationDbi")
# BiocManager::install("rrvgo")


library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(rrvgo)

library(purrr)
library(stringr)
library(ggplot2)
library(patchwork)
library(dplyr)


celltype <- 'exneu'
args = commandArgs(trailingOnly=TRUE)
cat(args)
ps_model <- as.character(args[1])
ps_model <- 'random_forest_cv'

path_rs_GO <- sprintf('~/../jinandmaya/AD/results/')
dir.create(path_rs_GO, recursive=TRUE, showWarnings = FALSE)

########################################################################################
#
# Cnetplot
#
########################################################################################
GO_results <- list()
methods <- c('causarray', 'ruv')
for (method in methods) {
    df_list <- list()
    for (dataset in c('ROSMAP-AD/results', 'SEA-AD/results-MTG', 'SEA-AD/results-PFC')){
        path_rs <- sprintf('~/../jinandmaya/%s/%s/DE/', dataset, ps_model)
        df <- read.csv(sprintf('%sres.%s.%s.csv', path_rs, celltype, method), row.names=1)
        if('tau' %in% colnames(df)){df$log2FoldChange <- df$tau / log(2)}
        df_list <- append(df_list, list(df))
    }
    # Get average log fold change for each gene
    foldChange <- sapply(df$gene_names, function(gene) {
        logfoldchanges <- sapply(df_list, function(df) {return(df$log2FoldChange[df$gene_names == gene])})
        median(logfoldchanges, na.rm = TRUE)
    })
    
    # Get overlap of discoveries
    significant_genes <- unique(unlist(lapply(df_list, function(df) df$gene_names[df$padj <= 0.1])))
    significant_genes_logfoldchange <- foldChange[significant_genes]
    
    # Select top 50 genes based on absolute log fold change
    genes_to_test <- c(names(sort(significant_genes_logfoldchange, decreasing = TRUE))[1:25],
        names(sort(significant_genes_logfoldchange, decreasing = FALSE))[1:25])
    # genes_to_test <- names(significant_genes_logfoldchange)
    length(genes_to_test)
    
    
    #Run groupGO using the names of the vector
    GO_results[[method]] <- groupGO(
           gene     = genes_to_test,
           OrgDb    = org.Hs.eg.db, keyType = "SYMBOL",
           ont      = 'BP',
           level    = 3,
           readable = TRUE)
    
    GO_results[[method]]@result <- GO_results[[method]]@result[GO_results[[method]]@result$Count>0,]
    GO_results[[method]]@result <- GO_results[[method]]@result[order(-GO_results[[method]]@result$Count),]
    write.csv(GO_results[[method]]@result, sprintf("%sGO_%s.csv", path_rs_GO, method))
    
    #Plot your results
    p <- cnetplot(GO_results[[method]], showCategory=GO_results[[method]]@result$Description[1:5],
             categorySize="GeneNum",
             circular = TRUE, colorEdge = TRUE, cex_label_gene = 1,
             color.params = list(foldChange = foldChange), color_category = "#309630",
             order=TRUE) + theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
    
    ggsave(sprintf("%scnetplot_%s.pdf", path_rs_GO, method), plot = p, width = 10, height = 8, units = "in")
}
