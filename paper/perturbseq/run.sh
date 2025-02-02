## Preprocessing
Rscript --no-save --no-restore --verbose 1-preprocess.R > out.txt 2>&1 
mv out.txt SCP1184/out.txt

## DE test
Rscript --no-save --no-restore --verbose 2-DE.R > out.txt 2>&1 
mv out.txt results/DE/out.txt

## GO analysis
Rscript --no-save --no-restore --verbose 3-GO.R > out.txt 2>&1 
mv out.txt results/GO/out.txt