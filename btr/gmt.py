import os
import csv


class GMT(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.gmt = []
        self.suffix = ''
        self._read_gmt()

    def generate(self, dataset_genes=None):
        for inst in self.gmt:
            link = inst[0]
            desc = inst[1]
            gene_list = inst[2:]
            missing_list = []
            if dataset_genes:
                gene_list, missing_list = gene_list_intersect(gene_list,
                                                              dataset_genes)
            yield(link, desc, gene_list, missing_list)

    def _read_gmt(self):
        """ given a filepath, reads the gmt or txt file at that location, returning
        a list that can be used in the scripts
        """
        path = self.filepath
        if os.path.isfile(path):
            gmt = []
            with open(path) as f:
                rd = csv.reader(f, delimiter="\t", quotechar='"')
                for row in rd:
                    gmt.append(row)
            gmt = _standardize_gmt(gmt)
            gmt_suffix = path.split('/')[-1][:-4]
        else:
            files = os.listdir(path)
            gmt = []
            gmt_suffix = path.split('/')[-2]
            for f in files:
                print(f)
                with open(path + f) as fd:
                    rd = csv.reader(fd, delimiter="\t", quotechar='"')
                    gene_list = []
                    for row in rd:
                        gene_list.append(row[0])
                    gmt.append([f, 'user defined'] + gene_list)
        self.gmt = gmt
        self.suffix = gmt_suffix


def gene_list_intersect(gmt_genes, dataset_genes):
    """ return the intersection between the current gene list HGNC symbols and
    the columns in the dataset. return a second list, `missing` for any genes
    that are missing.
    """
    intersect = [x for x in gmt_genes if x in dataset_genes]
    missing = [x for x in gmt_genes if x not in dataset_genes]
    return intersect, missing


def _standardize_gmt(gmt):
    """ Takes a loaded list from a .gmt file and reformats it,
    if necessary, so that the html id is always at index 0 and the
    description is at index 1
    """
    if 'http' in gmt[0][1]:
        gmt_standard = [[x[1]] + [x[0]] + x[2:] for x in gmt]
    else:
        gmt_standard = gmt
    return gmt_standard
