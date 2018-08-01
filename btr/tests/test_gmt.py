from btr.gmt import GMT

folder_path = "test_data/hypotheses/synthetic/"
reactome_path = "test_data/hypotheses/PathwayCommons9.reactome.hgnc.gmt"


def test_load_folder():
    path = folder_path
    gmt = GMT(path)
    assert (len(gmt.gmt) == 4)


def test_load_file():
    path = reactome_path
    gmt = GMT(path)
    assert (len(gmt.gmt) == 3)


def test_generate():
    gmt_paths = [folder_path, reactome_path]
    for path in gmt_paths:
        gmt = GMT(path)
        for link, desc, gene_list, missing_list in gmt.generate():
            assert(len(link) > 0)
            assert(len(desc) > 0)
            assert(len(gene_list) > 0)
            assert(len(missing_list) == 0)


def test_generate_with_missing():
    gmt = GMT(reactome_path)
    all_genes = []
    missing_genes = []
    for g in gmt.gmt:
        all_genes += g[2:]
        missing_genes += g[2:7]
        all_genes = list(set(all_genes))
    all_genes = [x for x in all_genes if x not in missing_genes]
    for generated in gmt.generate(dataset_genes=all_genes):
        link, desc, gene_list, missing_list = generated
        assert(len(link) > 0)
        assert(len(desc) > 0)
        assert(len(gene_list) > 0)
        print(len(missing_list))
        assert(len(missing_list) == 5)
