for isimu in  '_d_2_r_1_noise_1.0'
do
    python simu_poi_data.py ${isimu}
    cp simu_poi_data.py data/simu_100${isimu}/
    cp simu_poi_fit.R data/simu_100${isimu}/
    Rscript --no-save --no-restore --verbose simu_poi_fit.R ${isimu} &> out${isimu}_fit.txt
    mv out${isimu}_fit.txt results/
    python simu_poi_plot.py ${isimu} &> out${isimu}_plot.txt
    mv out${isimu}_plot.txt results/
done