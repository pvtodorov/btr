# amp-ad: Analysis of AMP-AD datasets

## Preliminaries
To execute R scripts in this repository, you will need to pre-install several R packages. Since some of these packages come from Bioconductor, the easiest thing to do is to run the following two commands inside R:

    source("https://bioconductor.org/biocLite.R")
    biocLite( c("tidyverse","stringr","synapseClient","EnsDb.Hsapiens.v86","ensembldb") )
    
(Go get a cup of coffee...)

## Wrangling data
The datasets used by the analyses in this repository can be easily downloaded using command-line scripts. To download Mount Sinai Brain Bank (MSBB) dataset, run the following on the command line:

    Rscript msbb.R <path to data>

If `<path to data>` is left blank, then `/data/AMP-AD/MSBB` will be used by default.

## Set up a python env
Make sure `virtualenv` is installed. If not, install it via pip.
    
    pip install virtualenv

Make a new virtualenv. I named my ampad-env
    
    virtualenv ampad-env

Activate the env
    
    activate ampad-env

Install the packages we need
    
    pip install -r requirements.txt


## Obtain background predictions
We will use a random forest regression to predict the Braak score of of samples
based on the gene expression profile in the BM36 region. This region was picked
because it has multiple stages of the disease. The background will be generated
for gene sets of 10 to 1000, moving in steps of 10 randomly selected genes, where models will be
trained on the data and the R2 out-of-bag score will be recorded as the performance.
This will be repeated at least 10,000 times.

To perform this, run the `predict_background.py` script in the command line.
The script will run for the specified number of repetitions and dump a csv file
for each one.

    python predict_background.py <settings.json> <number of repetitions>


## Aggregate background runs
The repeated predictions will need to be aggregated into a single file.

    python aggregate_predictions <settings.json>


## Predict the preformance of gene sets
To predict the performance of a gene set, obtain the necessary .gmt file or
create a folder with .txt files which contain gene sets formatted as HGNC ID on
each line. These can then be supplied to the `predict_set.py` script as follows:

    python predict_set.py <settings.json> <path to gmt file or folder of txts>

This will output a csv file with the name `scores_<gmt_file or folder suffix>.csv`
The output will contain an identifier for the set, a description, the number of genes
in the set, the number of genes which were actually used (if there is incomplete
overlap between the dataset and the gene set not all genes are used), the R2 value,
the p_value, and an Benjamini-Hochberg adjusted p_value.

