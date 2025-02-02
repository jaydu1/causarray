library(Seurat)

counts <- read.csv('SCP1184/expression/expression_PertCortex.new.txt', sep='\t')

metadata <- read.csv('SCP1184/metadata/meta_PertCortex.txt', sep='\t')
metadata <- metadata[-1,]

counts <- counts[-1,]
genenames <- counts[,1]
counts <- counts[,-1]
rownames(counts) <- genenames

data <- CreateSeuratObject(counts = counts, meta.data = metadata)
saveRDS(object = data, file = "SCP1184/seurat-data.rds")