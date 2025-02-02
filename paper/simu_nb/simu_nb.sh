for isimu in '_d_1_r_4_noise_0.1' '_d_1_r_4_noise_0.2' '_d_1_r_4_noise_0.3'
do
    Rscript simu_nb_data.R ${isimu}
    cp simu_nb_data.R data/simu_100${isimu}/
    cp simu_nb_fit.R data/simu_100${isimu}/
    Rscript --no-save --no-restore --verbose simu_nb_fit.R ${isimu} &> out${isimu}_fit.txt
    mv out${isimu}_fit.txt results/
    
    python simu_nb_plot.py ${isimu} &> out${isimu}_plot.txt
    mv out${isimu}_plot.txt results/
done