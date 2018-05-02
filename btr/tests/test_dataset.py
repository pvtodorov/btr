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


def load_dataset_filtered_AC(filter_dataset=True):
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
    dataset = Dataset(settings, filter_dataset=filter_dataset)
    return dataset, settings


def load_dataset_digitized(transform_dataset=True):
    settings = {"dataset": {
        "name": "synthetic_test_data",
        "filepath": "~/Code/btr/btr/tests/test_data/synthetic.tsv",
        "meta_columns": ["ID", "PMI", "AOD", "CDR",
                         "Braak", "BrodmannArea", "Barcode"],
        "target": "Braak",
        "ID_column": "ID",
        "transform": {"digitize": [{"column": "Braak",
                                    "thresholds": [0, 3, 5]}]}}}
    dataset = Dataset(settings, transform_dataset=transform_dataset)
    return dataset, settings


dataset_loads = [load_dataset_basic,
                 load_dataset_filtered_AC,
                 load_dataset_digitized]


def test_dataset_name():
    for f in dataset_loads:
        dataset, settings = f()
        name = settings['dataset']['name']
        assert(dataset.name == name)


def test_dataset_filepath():
    for f in dataset_loads:
        dataset, settings = f()
        filepath = settings['dataset']['filepath']
        assert(dataset.filepath == filepath)


def test_dataset_meta_columns():
    for f in dataset_loads:
        dataset, settings = f()
        assert(dataset.meta_cols == settings['dataset']['meta_columns'])


def test_dataset_data_columns():
    for f in dataset_loads:
        dataset, settings = f()
        data_columns = [x for x in dataset.data.columns.tolist()
                        if x not in settings['dataset']['meta_columns']]
        assert(data_columns == dataset.data_cols)


def test_dataset_target_column():
    for f in dataset_loads:
        dataset, settings = f()
        target = settings['dataset']['target']
        assert(dataset.target == target)


def test_dataset_id_column():
    for f in dataset_loads:
        dataset, settings = f()
        id_col = settings['dataset']['ID_column']
        assert(dataset.id_col == id_col)


def test_dataframe_length():
    for f in dataset_loads:
        dataset, settings = f()
        assert(len(dataset.data) > 0)


def test_dataframe_no_nan():
    for f in dataset_loads:
        dataset, settings = f()
        assert(dataset.data.isnull().any().sum() == 0)


def test_target_unique_values_before_filter():
    dataset, settings = load_dataset_basic()
    sorted_target_values = sorted(set(dataset.data[dataset.target]))
    sorted_expected_values = [0, 1, 2, 3, 4, 5, 6]
    assert(sorted_target_values == sorted_expected_values)


def test_filter_from_settings_during_load():
    dataset, settings = load_dataset_filtered_AC(filter_dataset=True)
    expected_values = settings['dataset']['filter']['filters'][0]['values']
    actual_values = dataset.data[dataset.target].unique().tolist()
    actual_values = sorted(actual_values)
    assert(actual_values == expected_values)


def test_filter_from_settings_after_load():
    dataset, settings = load_dataset_filtered_AC(filter_dataset=False)
    actual_values = sorted(dataset.data[dataset.target].unique().tolist())
    expected_values = [0, 1, 2, 3, 4, 5, 6]
    assert(actual_values == expected_values)
    expected_values = settings['dataset']['filter']['filters'][0]['values']
    dataset.filter_dataset()
    actual_values = sorted(dataset.data[dataset.target].unique().tolist())
    assert(actual_values == expected_values)


def test_filter_not_from_settings():
    dataset, settings = load_dataset_basic()
    expected_values = [0, 1, 2, 5, 6]
    filters = [{"column": "Braak",
                "values": expected_values}]
    dataset.filter_dataset(filters=filters)
    actual_values = sorted(dataset.data[dataset.target].unique().tolist())
    assert(actual_values == expected_values)


def test_digitized_from_settings_during_load():
    dataset, settings = load_dataset_digitized(transform_dataset=True)
    expected_values = [0, 1, 2]
    transformed_values = sorted(dataset.data[dataset.target].unique().tolist())
    assert(transformed_values == expected_values)


def test_digitized_from_settings_after_load():
    dataset, settings = load_dataset_digitized(transform_dataset=False)
    expected_values = [0, 1, 2, 3, 4, 5, 6]
    transformed_values = sorted(dataset.data[dataset.target].unique().tolist())
    assert(transformed_values == expected_values)
    dataset.transform_dataset()
    expected_values = [0, 1, 2]
    transformed_values = sorted(dataset.data[dataset.target].unique().tolist())
    assert(transformed_values == expected_values)


def test_digitized_after_load_not_from_settings():
    dataset, settings = load_dataset_basic()
    expected_values = [0, 1, 2, 3, 4, 5, 6]
    transformed_values = sorted(dataset.data[dataset.target].unique().tolist())
    assert(transformed_values == expected_values)
    transform = {"digitize": [{"column": "Braak",
                               "thresholds": [0, 3, 5]}]}
    dataset.transform_dataset(transform=transform)
    expected_values = [0, 1, 2]
    transformed_values = sorted(dataset.data[dataset.target].unique().tolist())
    assert(transformed_values == expected_values)


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
