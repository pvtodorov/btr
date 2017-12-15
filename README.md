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
