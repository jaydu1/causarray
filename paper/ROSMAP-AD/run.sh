
for ps_model in "random_forest_cv"
do
    # DE test
    Rscript --no-save --no-restore --verbose 2-DE.R ${ps_model} > out.txt 2>&1 
    mv out.txt results/${ps_model}/DE/out.txt

    # GO analysis
    Rscript --no-save --no-restore --verbose 3-GO.R ${ps_model} > out.txt 2>&1 
    mv out.txt results/${ps_model}/GO/out.txt

    python 4-CATE.py ${ps_model} > out.txt 2>&1
    mv out.txt results/${ps_model}/DE/out_CATE.txt
done