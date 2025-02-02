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
dataset <- as.character(args[2])

# ps_model <- 'random_forest_cv'
# dataset <- 'PFC'
path_rs <- sprintf('~/../jinandmaya/SEA-AD/results-%s/%s/DE/', dataset, ps_model)
path_rs_GO <- sprintf('~/../jinandmaya/SEA-AD/results-%s/%s/GO/', dataset, ps_model)
dir.create(path_rs_GO, recursive=TRUE, showWarnings = FALSE)

method <- 'ruv'
df_deseq <- read.csv(sprintf('%sres.%s.%s.csv', path_rs, celltype, method), row.names=1)
sum(df_deseq$padj<0.1)

method <- 'causarray'
df_causarray <- read.csv(sprintf('%sres.%s.%s.csv', path_rs, celltype, method), row.names=1)
# sum(df_causarray$rej)
sum(df_causarray$padj<0.1)






# Function to create a dot plot for GO terms
create_GO_dotplot <- function(GO_results, save_plot = FALSE, 
                              show = 'shared', top_n = 10, order_by = "pvalue", title = NULL,
                              plot_width = 8, plot_height = 6, plot_dpi = 300, 
                              y_text_size = 10, y_text_angle = 0, y_margin = NULL, wrap_width = 25) {

    # Extract result data frames
    causarray_results <- GO_results[['causarray']]@result
    deseq_results <- GO_results[['ruv']]@result
    
    # Find common GO terms
    common_GO_terms <- intersect(causarray_results$Description, deseq_results$Description)
    
    if (show == 'shared') {
        
        causarray_results <- causarray_results[causarray_results$Description %in% common_GO_terms, ]
        deseq_results <- deseq_results[deseq_results$Description %in% common_GO_terms, ]
        
        # Combine results
        combined_results <- bind_rows(
            causarray_results %>% mutate(Method = "causarray"),
            deseq_results %>% mutate(Method = "ruv")
        )
    } else if(show == 'causarray') {
        # Find distinct GO terms
        combined_results <- causarray_results[!causarray_results$Description %in% deseq_results$Description, ]
    } else {
        combined_results <- deseq_results[!deseq_results$Description %in% causarray_results$Description, ]
    }
    
    # Calculate necessary columns
    combined_results <- combined_results %>%
        mutate(GeneRatio_num = as.numeric(sub("/.*", "", GeneRatio)) / as.numeric(sub(".*/", "", GeneRatio)),
               `p.adjust` = p.adjust,
               Count = Count) %>%
        group_by(Description) %>%
        summarise(GeneRatio = max(GeneRatio_num),
                  Count = sum(Count),
                  `p.adjust` = min(`p.adjust`)) %>%
        ungroup()
    
    # Select top N terms
    if (order_by == "pvalue") {
        top_results <- combined_results %>% arrange(`p.adjust`) %>% head(top_n)
    } else if (order_by == "count") {
        top_results <- combined_results %>% arrange(desc(Count)) %>% head(top_n)
    }
    
    if (is.na(title)) {
        title <- sprintf("Top %s GO Terms", str_to_title(show))
    }
    top_results$Description <- str_wrap(top_results$Description, width = wrap_width)

    # Create dot plot
    p <- ggplot(top_results, aes(x = GeneRatio, y = reorder(Description, Count), size = Count, color = `p.adjust`)) +
        geom_point(alpha = 0.7) +
        scale_size_continuous(range = c(3, 10)) +
        scale_color_gradient(low = "blue", high = "red") +
        labs(x = "GeneRatio", y = "GO Terms", size = "Count", color = "p.adjust", 
             title = title) +
        theme_minimal() +
        theme(
            axis.text.y = element_text(size = y_text_size, angle = y_text_angle),
            axis.text.x = element_text(size = 10),
            plot.margin = if (!is.null(y_margin)) margin(y_margin[1], y_margin[2], y_margin[3], y_margin[4], "pt"),
            plot.title = element_text(size=16),
            legend.text = element_text(size=8),
            legend.title = element_text(size = 8),
            legend.key.size = unit(0.5, "lines"),   # Adjust legend key size
            legend.key.width = unit(0.5, "lines"),  # Adjust legend key width
            legend.box.spacing = unit(0.5, "lines") # Adjust spacing between legend elements
        ) +
        guides(
            size = guide_legend(override.aes = list(size = 2), order = 1),
            color = guide_colorbar(barwidth = 0.5, barheight = 5, order = 2)
        )
    
    # Save the plot if specified
    if (save_plot != FALSE) {
        ggsave(save_plot, plot = p, width = plot_width, height = plot_height, dpi = plot_dpi)
        write.csv(top_results, file=str_replace(save_plot, '.pdf', '.csv'))
    }
    
    cat(top_results$Description)
    
    # Return the plot object
    return(p)
}


# celltype <- 'exneu'
# path_rs <- sprintf('~/../jinandmaya/pmt/results/pmt/%s/', ps_model)

# Read and perform GO enrichment analysis for each method
GO_results <- list()
methods <- c('causarray', 'ruv')
for (method in methods) {
    df <- read.csv(sprintf('%sres.%s.%s.csv', path_rs, celltype, method), row.names=1)
    # genes_to_test <- sort(df[(df$padj <= 0.1), ]$gene_names)
    # GO_results[[method]] <- enrichGO(
    #     gene = genes_to_test, OrgDb = org.Hs.eg.db, keyType = "SYMBOL", ont = "BP")

    genes_to_test <- sort(df[(df$padj <= 0.1), ]$ensembl_gene_id)
    GO_results[[method]] <- enrichGO(
        gene = genes_to_test, OrgDb = org.Hs.eg.db, keyType = "ENSEMBL", ont = "BP")
    GO_results[[method]] <- setReadable(GO_results[[method]], OrgDb = org.Hs.eg.db)

    all_GO_results <- as.data.frame(GO_results[[method]])
    cat(method, length(genes_to_test), dim(all_GO_results)[1], "\n")
    write.csv(all_GO_results, sprintf("%sGO_%s.csv", path_rs_GO, method))

    GO_results[[method]]@result <- GO_results[[method]]@result[
        (GO_results[[method]]@result$p.adjust<0.05) & (GO_results[[method]]@result$qvalue<0.2),]
}

# To show shared GO terms
p_list <- list()
title_list <- list(
  'shared' = "shared",
  'causarray' = "causarray",
  'ruv' = "RUV"
)
for (name in c("shared", "causarray", "ruv")){
    cat(name,"\n")
    p_list[[name]] <- create_GO_dotplot(
        GO_results, sprintf("%sGO_top_%s.pdf", path_rs_GO, name), name,
        top_n = 10, order_by = "count", title=title_list[[name]],
        plot_width = 8, plot_height=6,
        y_text_size = 9, y_text_angle = 0, y_margin = c(5, 5, 20, 5))
    cat('\n')
}

p <- p_list$shared + plot_spacer() + p_list$causarray + plot_spacer() + p_list$ruv + 
    plot_layout(widths = c(4, -.5 ,4.5, -.5 ,4.5), nrow = 1, axis_titles = "collect")
ggsave(sprintf("%sGO_%s.pdf", path_rs_GO, 'all'), plot = p, width = 18, height = 6, dpi = 300)