



# Requirements

The code in this repository requires the following softwares to be installed.
We recommend using the [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager to install the necessary packages.

```cmd
# create a new conda environment and install the necessary packages
mamba create -n renv r-essentials r-base -y
# activate the environment
mamba activate renv

# install the necessary Python packages
mamba install -c conda-forge numba pandas numpy scipy statsmodels scikit-learn cvxpy tqdm scanpy -y

# install the necessary R packages
mamba install -c conda-forge -c bioconda r-devtools r-reticulate r-rcpp r-rcppeigen r-rcppprogress r-dqrng r-sitmo r-bh r-matrix r-seurat r-seuratobject r-doparallel r-qs bioconductor-deseq2 bioconductor-qvalue bioconductor-rhdf5 r-anndata -y

# extra R packages required for RUV-III-NB
mamba install -c conda-forge -c bioconda bioconductor-edger r-rfast bioconductor-singlecellexperiment r-desctools bioconductor-scater bioconductor-scran bioconductor-scuttle bioconductor-singler bioconductor-celldex r-hrbrthemes r-randomcolor bioconductor-dittoseq r-bookdown bioconductor-delayedmatrixstats bioconductor-delayedarray bioconductor-splatter bioconductor-hdf5array -y


# RUV-seq
mamba install conda-forge::r-ggpubr conda-forge::r-meta bioconda::bioconductor-reactomepa -y
mamba install r-zim -y
mamba install bioconda::bioconductor-ruvseq bioconda::bioconductor-zebrafishrnaseq bioconda::bioconductor-muscat -y

# for GO analysis
mamba install -c conda-forge -c bioconda bioconductor-clusterprofiler bioconductor-org.hs.eg.db bioconductor-annotationdbi bioconda::bioconductor-rrvgo -y


# install the necessary Python packages
pip install scib cinemaot
```

The cocoA-diff and RUV-III-NB packages need to be installed via the following commands in R:
```r
# install the cocoA-diff package in R
devtools::install_github("causalpathlab/mmutilR@main", dependencies=F)

# install the RUV-III-NB package in R
devtools::install_github("limfuxing/ruvIIInb", dependencies=F, build_vignettes=FALSE)

```

One can also use the yml file to create the environment:
```cmd
mamba env create -f environment.yml
```
Note that the yml file specifies the version of the packages that were used to generate the results in the paper, which may not be applicable to different operating systems.
The results may vary if the versions of the packages are different.


# Reproducibility workflow

## Simulation with bulk-cell data

The code for simulation with bulk-cell data is in the folder `simu_poi`.
To execute the scripts from the command line, use the following commands:
```cmd
cd simu_poi
bash run.sh
```
Then one can use Jupyter notebook `Plot.ipynb` to plot the results in the folder `simu_poi`.

## Simulation with single-cell data

The code for simulation with bulk-cell data is in the folder `simu_nb`.
To execute the scripts from the command line, use the following commands:
```cmd
cd simu_nb
bash run.sh
```
Then one can use Jupyter notebook `Plot.ipynb` to plot the results in the folder `simu_nb`.


## Case study with Perturb-seq data


The code for case study with erturb-seq data is in the folder `perturbseq`.
The data folder `perturbseq/SCP1184` can be downloaded from [url](https://singlecell.broadinstitute.org/single_cell/study/SCP1184/in-vivo-perturb-seq-reveals-neuronal-and-glial-abnormalities-associated-with-asd-risk-genes#/).


```cmd
cd perturbseq
bash run.sh
# use Plot.ipynb for plotting
cd ..
```

## Case study with AD data

The code for case study with AD data is in the folder `ROSMAP-AD`, `SEA-AD`, and `AD`.
There are three datasets in the folder: `ROSMAP-AD` and `SEA-AD` (MTG and PFC).
The results are further analyzed using code in the folder `AD`.

```cmd
cd ROSMAP-AD
bash run.sh

cd ../SEA-AD
bash run.sh

cv ../AD
Rscript GO.R
# use Plot.ipynb for plotting
cd ..
```


