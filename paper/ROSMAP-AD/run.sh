Rscript --no-save --no-restore --verbose 1-preprocess.R > out.txt 2>&1 
mv out.txt data/out.txt

for ps_model in "logistic" "random_forest_cv" 
do
    ## DE test
    Rscript --no-save --no-restore --verbose 2-DE.R ${ps_model} > out.txt 2>&1 
    mv out.txt results/${ps_model}/DE/out.txt

    ## GO analysis
    Rscript --no-save --no-restore --verbose 3-GO.R ${ps_model} > out.txt 2>&1 
    mv out.txt results/${ps_model}/GO/out.txt    
done

python 4-CATE.py > out.txt 2>&1
mv out.txt results/random_forest_cv/DE/out-cate.txt