from btr.dataset import Dataset


def load_dataset_basic():
    settings = {"dataset": {
        "name": "synthetic_test_data",
        "filepath": "~/Code/btr/btr/tests/test_data/synthetic.tsv",
        "meta_columns": ["ID", "PMI", "AOD", "CDR",
                         "Braak", "BrodmannArea", "Barcode"],
        "target": "Braak",
        "ID_column": "ID"}}
    dataset = Dataset(settings)
    print(dataset)
    return dataset, settings


def load_dataset_filtered_AC():
    values = [0, 1, 2, 5, 6]
    settings = {"dataset": {
        "name": "synthetic_test_data",
        "filepath": "~/Code/btr/btr/tests/test_data/synthetic.tsv",
        "meta_columns": ["ID", "PMI", "AOD", "CDR",
                         "Braak", "BrodmannArea", "Barcode"],
        "target": "Braak",
        "ID_column": "ID",
        "filter": {"name": "AC",
                   "filters": [{"column": "Braak",
                                "values": values}]},
        "transform": {}}}
    dataset = Dataset(settings)
    return dataset, settings


dataset, settings = load_dataset_basic()


def test_dataset_name():
    name = settings['dataset']['name']
    assert(dataset.name == name)


def test_dataset_filepath():
    filepath = settings['dataset']['filepath']
    assert(dataset.filepath == filepath)


def test_dataset_meta_columns():
    assert(dataset.meta_cols == settings['dataset']['meta_columns'])


def test_dataset_data_columns():
    data_columns = [x for x in dataset.data.columns.tolist()
                    if x not in settings['dataset']['meta_columns']]
    assert(data_columns == dataset.data_cols)


def test_dataset_target_column():
    target = settings['dataset']['target']
    assert(dataset.target == target)


def test_dataset_id_column():
    id_col = settings['dataset']['ID_column']
    assert(dataset.id_col == id_col)


def test_dataframe_length():
    assert(len(dataset.data) > 0)


def test_dataframe_no_nan():
    assert(dataset.data.isnull().any().sum() == 0)


def test_target_unique_values_before_filter():
    sorted_target_values = sorted(set(dataset.data[dataset.target]))
    sorted_expected_values = [0, 1, 2, 3, 4, 5, 6]
    assert(sorted_target_values == sorted_expected_values)


def test_filter_from_settings():
    dataset, settings = load_dataset_filtered_AC()
    filtered_values = sorted(dataset.data[dataset.target].unique().tolist())
    provided_values = settings['dataset']['filter']['filters'][0]['values']
    assert(filtered_values == provided_values)


def test_filter_after_load():
    dataset, settings = load_dataset_basic()
    provided_values = [0, 1, 2, 5, 6]
    filters = [{"column": "Braak",
                "values": provided_values}]
    dataset.filter_dataset(filters=filters)
    filtered_values = sorted(dataset.data[dataset.target].unique().tolist())
    assert(filtered_values == provided_values)


def test_sample_cols_noseed():
    dataset, settings = load_dataset_basic()
    for i in range(10, 110, 10):
        cols1 = dataset.sample_data_cols(i)
        cols2 = dataset.sample_data_cols(i)
        cols_overlap = [x for x in cols1 if x in cols2]
        assert(len(cols1) == len(cols2))
        assert(len(cols_overlap) != len(cols2))


def test_sample_cols_seeded():
    dataset, settings = load_dataset_basic()
    for i in range(10, 110, 10):
        cols1 = dataset.sample_data_cols(i, seed=47)
        cols2 = dataset.sample_data_cols(i, seed=47)
        cols_overlap = [x for x in cols1 if x in cols2]
        assert(len(cols1) == len(cols2))
        assert(len(cols_overlap) == len(cols1))
