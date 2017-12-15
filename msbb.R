## Wrangling of MSBB RNAseq data an matching clinical covariates
##
## by Artem Sokolov

suppressMessages(library( tidyverse ))
suppressMessages(library( synapseClient ))
library( stringr )

## Composes a mapping between ENSEMBL IDs and HUGO names
ens2hugo <- function()
{
    edb <- EnsDb.Hsapiens.v86::EnsDb.Hsapiens.v86
    tx <- ensembldb::transcripts( edb, column=c("gene_id", "gene_name") )
    data_frame( HUGO = tx$gene_name, ENSEMBL = tx$gene_id ) %>% distinct
}

## Parse local directory specification
argv <- commandArgs( trailingOnly = TRUE )
if( length(argv) == 0 )
{
    cat( "NOTE: No directory specified on command line. Using default.\n" )
    local.dir <- "/data/AMP-AD/MSBB"
} else { local.dir <- argv[1] }


## Create directory if it doesn't exist
dir.create( local.dir, showWarnings=FALSE )
cat( "Wrangling MSBB dataset to", local.dir, "\n" )

## Login to Synapse and download/wrangle data
cat( "Logging in to Synapse...\n" )
synapseLogin( rememberMe=TRUE )

## Read raw expression matrix
cat( "Downloading expression data...\n" )
fnX <- synGet( "syn7809023", downloadLocation = local.dir )@filePath
cat( "Loading local copy...\n" )
Xraw <- read.delim( fnX, check.names=FALSE )

## Map Gene IDs to HUGO
## Remove alternative splice forms and duplicated entries
## NOTE: There are only 42 genes with duplicate entries
##   If there were more, the correct approach would be to average their expression
##   across the duplicates, rather than dropping.
## Retrieve the sample barcode
cat( "Mapping gene IDs to HUGO...\n" )
E2H <- ens2hugo()
X <- Xraw %>% rownames_to_column( "ENSEMBL" ) %>% inner_join( E2H, ., by="ENSEMBL" ) %>%
    filter( !grepl( "\\.", HUGO ) ) %>% filter( !duplicated(HUGO) ) %>%
    select( -ENSEMBL ) %>% gather( barcode, Value, -HUGO )

## Match sample barcodes against individual IDs and brain region information
cat( "Annotating samples with brain region...\n" )
fnZ <- synGet( "syn6100548", downloadLocation = local.dir )@filePath
XZ <- suppressMessages( read_csv(fnZ) ) %>%
    select( BrodmannArea, barcode, individualIdentifier ) %>% distinct %>%
    mutate( barcode = as.character(barcode) ) %>% inner_join( X, by = "barcode" )

## Load clinical covariates and combine with the expression matrix
cat( "Downloading clinical covariates...\n" )
fnY <- synGet( "syn6101474", downloadLocation = local.dir )@filePath
XY <- suppressMessages( read_csv( fnY ) ) %>% select( individualIdentifier, CDR, bbscore ) %>%
    inner_join( XZ, by = "individualIdentifier" )

## Flatten the matrix to samples-by-(clin+genes) and save to file
cat( "Finalizing...\n" )
RR <- spread( XY, HUGO, Value ) %>%
    rename( ID = individualIdentifier, Braak = bbscore, Barcode = barcode )
fnOut <- file.path( local.dir, "msbb-wrangled.tsv" )
cat( "Writing output to", fnOut, "\n" )
write_tsv( RR, fnOut )

