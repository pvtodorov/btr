from btr.processing_schemes import PairsProcessor
from btr.tests.test_dataset import load_dataset_digitized_str2float_confounder

estimator_dict = {"name": "Multiclass_Linear",
                  "estimator_params": {}}
steps_dict = [{"operation": "contains_sample"},
              {"operation": "diff_target"},
              {"operation": "unique_pair"},
              {"operation": "min_confounder"}]
lpocv_settings = {"name": "LPOCV",
                  "pair_settings": {"shuffle_samples": True,
                                    "seed": 47,
                                    "steps": steps_dict},
                  "estimator": estimator_dict}


def test_pairs_proc_build_pairs_seed():
    dataset = load_dataset_digitized_str2float_confounder()
    for seed_pair in [(0, 0), (47, 47), (0, 47), (None, None)]:
        pairs_settings_1 = {"shuffle_samples": True,
                            "seed": seed_pair[0],
                            "steps": steps_dict}
        pairs_settings_2 = {"shuffle_samples": True,
                            "seed": seed_pair[1],
                            "steps": steps_dict}
        pairs_proc_1 = PairsProcessor(dataset=dataset,
                                      pair_settings=pairs_settings_1)
        pairs_proc_2 = PairsProcessor(dataset=dataset,
                                      pair_settings=pairs_settings_2)
        if (seed_pair[0] == seed_pair[1]) and (seed_pair[0] is not None):
            assert(pairs_proc_1.selected_pairs == pairs_proc_2.selected_pairs)
        else:
            assert(pairs_proc_1.selected_pairs != pairs_proc_2.selected_pairs)


def test_pairs_proc_build_pairs_unseed():
    dataset = load_dataset_digitized_str2float_confounder()
    pairs_settings_1 = {"shuffle_samples": True,
                        "steps": steps_dict}
    pairs_settings_2 = {"shuffle_samples": True,
                        "steps": steps_dict}
    pairs_proc_1 = PairsProcessor(dataset=dataset,
                                  pair_settings=pairs_settings_1)
    pairs_proc_2 = PairsProcessor(dataset=dataset,
                                  pair_settings=pairs_settings_2)
    assert(pairs_proc_1.selected_pairs != pairs_proc_2.selected_pairs)
