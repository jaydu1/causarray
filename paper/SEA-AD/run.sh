for dataset in "PFC" "MTG"
do
    ## Preprocessing
    Rscript --no-save --no-restore --verbose 1-preprocess.R ${dataset} > out-${dataset}.txt 2>&1 
    mv out-${dataset}.txt data/out-${dataset}.txt
    
    for ps_model in "random_forest_cv"
    do
        ## DE test
        Rscript --no-save --no-restore --verbose 2-DE.R ${ps_model} ${dataset} > out-${dataset}.txt 2>&1 
        mv out-${dataset}.txt results-${dataset}/${ps_model}/DE/out.txt

        ## GO analysis
        Rscript --no-save --no-restore --verbose 3-GO.R ${ps_model} ${dataset} > out-${dataset}.txt 2>&1 
        mv out-${dataset}.txt results-${dataset}/${ps_model}/GO/out.txt
    done
done