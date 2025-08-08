for isimu in '_d_1_r_4_noise_0.1' '_d_1_r_4_noise_0.2' '_d_1_r_4_noise_0.3'
do
    Rscript simu_nb_data.R ${isimu}
    cp simu_nb_data.R data/simu_100${isimu}/
    cp simu_nb_fit.R data/simu_100${isimu}/
    for seed in $(seq 0 49);
    do
        Rscript --no-save --no-restore --verbose simu_nb_fit.R ${isimu} ${seed} &> out${isimu}_fit_${seed}.txt
        mv out${isimu}_fit_${seed}.txt results/
    done
    python simu_nb_plot.py ${isimu} &> out${isimu}_plot.txt
    mv out${isimu}_plot.txt results/
done
