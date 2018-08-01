# Better than Random analysis

**Authors:** Petar Todorov<sup>1</sup>, Artem Sokolov<sup>1</sup><br />
<sup>1</sup>Laboratory of Systems Pharmacology, Harvard Medical School

## What is this?

This repo aims to offer an easy way to test gene set hypotheses against
backgrounds of randomly chosen gene sets predictions. Gene sets which are more
predictive than the distribution of randomly chosen gene sets may indicate a
link to the importance of their constituent genes.

## Prerequisites

We get started by creating a Python 3 virtual environment.

    virtualenv nameyourenvhere

In case you are setting this up on a cluster, you may want to use the packages
compiled by the cluster. To do that:

    virtualenv nameyourenvhere --system-site-packages

To activate your environment

    source nameyourenvhere/bin/activate

Then clone this repo, and install it as editable using pip

    git clone https://github.com/pvtodorov/btr.git
    cd btr
    pip install -e .

You're ready to go!


## Using the software

Installing the repo will also bind some commands that can be used in the terminal.
In order to specify how to run the software, a settings file is needed. An example
can be see in this repo's `example_settings.json`. If a background is being generated,
a file such as `example_background_params.json` must be provided. If a hypothesis is being
used as the feature set, a `.gmt` file must be used such as this [this](http://www.pathwaycommons.org/archives/PC2/v10/PathwayCommons10.reactome.uniprot.gmt.gz) from Pathway Commons. 

### Making predictions

To generate background predictions:

    btr-predict <path to settings file> -b <path to background parmeters>

To generate geneset predictions:

    btr-predict <path to settings file> -g <path to GMT file or folder with txt gene lists>


### Evaluating predictions

To evaluate predictions, first score the background runs

    btr-score <path to settings file>

To evaluate gene files, score them

    btr-score <path to settings file> -g <path to GMT file or folder with txt gene lists>


### Testing for statistical significance

    btr-stats <path to settings file> -g <path to GMT file or folder with txt gene lists>
