
library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(rrvgo)

library(purrr)
library(stringr)
library(ggplot2)
library(patchwork)
library(dplyr)



name <- 'exneu'
path_rs <- sprintf('~/../jinandmaya/perturbseq/results/DE/')

method <- 'ruv'
df_deseq <- read.csv(sprintf('%sres.%s.%s.csv', path_rs, name, method), row.names=1)
sum(df_deseq$padj<0.1)

method <- 'causarray'
df_causarray <- read.csv(sprintf('%sres.%s.%s.csv', path_rs, name, method), row.names=1)
# sum(df_causarray$rej)
sum(df_causarray$padj<0.1)

trts <- unique(df_causarray$trt)



plot_GO <- function(genes_to_test, filename_GO, filename_GO_tree,
                    barplot_width=8, barplot_height=8, heatmap_width=8, heatmap_height=4){
    genes_to_test <- toupper(genes_to_test)
    GO_results <- enrichGO(gene = genes_to_test, OrgDb = org.Hs.eg.db, keyType = "SYMBOL", ont = "BP")
    
    n_GO <- length(genes_to_test[which(unlist(map(genes_to_test, function(x){grepl(x, str_c(GO_results$geneID, collapse = '/'))})))])
    if(n_GO==0){
        cat('No GO terms.', '\n')
        return()
    }
    cat(n_GO, '\n')
    
    p <- barplot(GO_results, showCategory = 10)
    pdf(filename_GO, width = barplot_width, height = barplot_height)
    plot(p)
    dev.off()
    
    possibleError <- tryCatch(
        {
    go_analysis <- as.data.frame(GO_results)
    simMatrix <- calculateSimMatrix(go_analysis$ID,
                                    orgdb="org.Hs.eg.db",
                                    ont="BP",
                                    method="Rel")
    scores <- setNames(-log10(go_analysis$p.adjust), go_analysis$ID)
    reducedTerms <- reduceSimMatrix(simMatrix,
                                    scores,
                                    threshold=0.7,
                                    orgdb="org.Hs.eg.db")
    
    pdf(filename_GO_tree, width = heatmap_width, height = heatmap_height)
    treemapPlot(reducedTerms)
    dev.off()
    },
    error=function(e) e
    )
}


res <- list()
for(method in c('causarray', 'ruv', 'deseq')){
    res[[method]] <- read.csv(sprintf('%s/res.%s.%s.csv', path_rs, name, method), row.names=1)
}


path_rs <- sprintf('~/../jinandmaya/perturbseq/results/')
for(trt in trts){
for(method in c('causarray', 'ruv', 'deseq')){
    idx <- (res[[method]]$padj<=0.1) & (res[[method]]$trt == trt)
    genes_to_test <- sort(
        res[[method]][idx,]$gene_names
    )
    cat(method, trt, length(genes_to_test), '\n')
    plot_GO(genes_to_test, sprintf("%sGO/%s_GO_%s.pdf", path_rs, trt, method), sprintf("%sGO/%s_GO_treemap_%s.pdf", path_rs, trt, method))
}
}



# Count the number of DE genes and GO terms
results_df <- data.frame(trt = character(), method = character(), n_DE_genes = integer(), n_GO_genes = integer(), n_GOs = integer(), stringsAsFactors = FALSE)

for(trt in trts){
for(method in c('causarray', 'ruv', 'deseq')){
    idx <- (res[[method]]$padj<=0.1) & (res[[method]]$trt == trt)
    genes_to_test <- sort(
        res[[method]][idx,]$gene_names
    )
    cat(method, trt, length(genes_to_test), '\n')
    genes_to_test <- toupper(genes_to_test)
    GO_results <- enrichGO(gene = genes_to_test, OrgDb = org.Hs.eg.db, keyType = "SYMBOL", ont = "BP")

    n_GO_genes <- length(genes_to_test[which(unlist(map(genes_to_test, function(x){grepl(x, str_c(GO_results$geneID, collapse = '/'))})))])
    n_GOs <- length(GO_results$ID)

    results_df <- rbind(results_df, data.frame(trt = trt, method = method, n_DE_genes=length(genes_to_test), n_GO_genes = n_GO_genes, n_GOs = n_GOs, stringsAsFactors = FALSE))
}
}
write.csv(results_df, sprintf("%sGO/GO.csv", path_rs))



################################################################################
# Selective results
# results_df <- read.csv(sprintf("%sGO.csv", path_rs))
GO_trts <- results_df[results_df$n_GO_genes>=275,'trt']

method <- 'causarray'

all_GO_results <- list()

# Collect all GO results
for (trt in GO_trts) {
    idx <- (res[[method]]$padj <= 0.1) & (res[[method]]$trt == trt)
    genes_to_test <- sort(res[[method]][idx,]$gene_names)
    cat(method, trt, length(genes_to_test), '\n')
    
    genes_to_test <- toupper(genes_to_test)
    GO_results <- enrichGO(gene = genes_to_test, OrgDb = org.Hs.eg.db, keyType = "SYMBOL", ont = "BP")
    all_GO_results[[trt]] <- GO_results
}

# Determine the shared color scale
all_pvalues <- unlist(lapply(all_GO_results, function(x) x$p.adjust))
shared_color_scale <- scale_fill_continuous(low = "red", high = "blue", name = "p.adjust", limits = range(all_pvalues), guide = guide_colorbar(reverse = TRUE))

# Create and save the plots
for (trt in GO_trts) {
    GO_results <- all_GO_results[[trt]]

    p <- barplot(GO_results, showCategory = 10) + 
        geom_bar(stat = "identity", width = 0.5) +
        theme(legend.position = "top", legend.title = element_text(hjust = 0.5), legend.text = element_text(size = 6))
    pdf(sprintf("%s%s_GO_%s.pdf", path_rs, trt, method), width = 5, height = 6)
    plot(p)
    dev.off()
    # plot_GO(genes_to_test, sprintf("%s%s_GO_%s.pdf", path_rs, trt, method), sprintf("%s%s_GO_treemap_%s.pdf", path_rs, trt, method),
    #         barplot_width=6, barplot_height=6)
}




################################################################################
dir.create(sprintf("%sGO_terms", path_rs), recursive=TRUE, showWarnings = FALSE)
all_GO_results <- list()
res <- list()
for(method in c('causarray', 'ruv', 'deseq')){
    res[[method]] <- read.csv(sprintf('%sDE/res.%s.%s.csv', path_rs, name, method), row.names=1)

    for (trt in trts) {
        idx <- (res[[method]]$padj <= 0.1) & (res[[method]]$trt == trt)
        genes_to_test <- sort(res[[method]][idx,]$gene_names)
        cat(method, trt, length(genes_to_test), '\n')
        
        genes_to_test <- toupper(genes_to_test)
        GO_results <- enrichGO(gene = genes_to_test, OrgDb = org.Hs.eg.db, keyType = "SYMBOL", ont = "BP")
        all_GO_results[[trt]] <- as.data.frame(GO_results)
        # write.csv(all_GO_results[[trt]], sprintf("%sGO_terms/%s_GO_%s.csv", path_rs, trt, method))
    }
}






################################################################################
# 
# Enriched GO terms under Satb2 perturbation
#
################################################################################
trt <- 'Satb2'
all_GO_results <- list()
for(method in c('causarray', 'ruv')){
    all_GO_results[[method]] <- read.csv(sprintf("%sGO_terms/%s_GO_%s.csv", path_rs, trt, method))
}

# Extract top 10 GO terms for each method
top_GO_results <- lapply(all_GO_results, function(df) {
  df <- df[order(df$p.adjust), ]  # Sort by p.adjust
  head(df, 10)  # Select top 10
})

# Combine the data frames into one for plotting
combined_GO_results <- do.call(rbind, lapply(names(top_GO_results), function(method) {
  df <- top_GO_results[[method]]
  df$Method <- method
  df
}))

# Plot
# Create individual plots for each method
custom_colors <- c("#e06663", "#327eba")
max_count <- max(sapply(top_GO_results, function(df) max(df$Count)))

plots <- lapply(names(top_GO_results), function(method) {
    df <- top_GO_results[[method]]
    df$log_p_adjust <- -log10(df$p.adjust)  # Transform p.adjust to -log10(p.adjust)
    ggplot(df, aes(x = reorder(Description, Count), y = Count, fill = p.adjust)) +
        geom_bar(stat = "identity") +
        scale_fill_gradient(low = custom_colors[1], high = custom_colors[2],  labels = scientific_format(digits = 2)) +
        coord_flip() +
        theme_minimal() +
        ylim(0, max_count) + 
        labs(title = paste("Top 10 GO Terms -", method),
             x = "GO Term",
             y = "Count",
             fill = "p.adjust")
})

# Arrange the plots side by side
combined_plot <- plots[[1]] / plots[[2]] + plot_layout(guides = 'collect')

ggsave(sprintf("%s%s_GO_Terms_Barplot.pdf", path_rs, trt), plot = combined_plot, width = 8, height = 12)