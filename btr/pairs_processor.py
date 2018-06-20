from itertools import combinations
import numpy as np


class PairsProcessor(object):
    """Returns pairs sampled for LPOCV
    """

    def __init__(self, dataset=None, pair_settings=None):
        self.selected_pairs = []
        if dataset:
            self.generate_all_pairs(dataset)
        if pair_settings:
            self.select_pairs(pair_settings)

    def generate_all_pairs(self, dataset):
        self._all_pairs_list = get_all_pairs(dataset)

    def select_pairs(self, pair_settings):
        """Generates reproducible list of pairs to use for LPOCV.
        """
        self.selected_pairs = []
        self._sample_list = get_sample_list(self._all_pairs_list)
        self._used_ids = {x: 0 for x in self._sample_list}
        self._seed = pair_settings.get("seed")
        self._prng = np.random.RandomState(self._seed)
        self._sample_list.reverse()
        steps = pair_settings['steps']
        while len(self._sample_list):
            self._sample = self._sample_list.pop()
            self._pairs_list_f = [x for x in self._all_pairs_list]
            for st in steps:
                self._perform_step(st)
            self.selected_pairs = self.selected_pairs + self._pairs_list_f
            for pair_id in get_sample_list(self._pairs_list_f):
                self._used_ids[pair_id] = self._used_ids[pair_id] + 1
        self._clean_up_attributes()

    def _perform_step(self, step):
        operation = step['operation']
        if operation == "contains_sample":
            self._pairs_list_f = _pair_op_contains_sample(self._pairs_list_f,
                                                          self._sample)
        elif operation == "diff_target":
            self._pairs_list_f = _pair_op_diff_target(self._pairs_list_f)
        elif operation == "unique_pair":
            self._pairs_list_f = _pair_op_unique_pair(self._pairs_list_f,
                                                      self.selected_pairs)
        elif operation == "unique_ids":
            self._pairs_list_f = _pair_op_unique_ids(self._pairs_list_f,
                                                     self._used_ids)
        elif operation == "min_id_reuse":
            self._pairs_list_f = _pair_op_min_id_reuse(self._pairs_list_f,
                                                       self._used_ids)
        elif operation == "min_confounder":
            self._pairs_list_f = _pair_op_min_confounder(self._pairs_list_f)
        elif operation == "pick_one":
            self._pairs_list_f = _pair_op_pick_one(self._pairs_list_f,
                                                   self._prng)
        elif operation == "take_all":
            self._pairs_list_f = self._pairs_list_f

    def _clean_up_attributes(self):
        del self._sample_list
        del self._used_ids
        del self._seed
        del self._prng
        del self._sample
        del self._pairs_list_f


def _pair_op_contains_sample(pairs_list, sample):
    """Returns pairs_list that contains the sample"""
    pairs_list_f = [x for x in pairs_list
                    if sample in [x[0][0], x[1][0]]]
    return pairs_list_f


def _pair_op_diff_target(pairs_list):
    """Return pairs list filtered so target value is different"""
    pairs_list_f = [x for x in pairs_list
                    if x[0][1] != x[1][1]]
    return pairs_list_f


def _pair_op_unique_pair(pairs_list, used_pairs):
    """Return pairs_list that does not contain previously used pairs"""
    pairs_list_f = [x for x in pairs_list
                    if x not in used_pairs]
    return pairs_list_f


def _pair_op_unique_ids(pairs_list, used_ids):
    """Return pairs_list that does not contain previously used ids"""
    pairs_list_f = [x for x in pairs_list
                    if ((used_ids[x[0][0]] == 0) and
                        (used_ids[x[1][0]] == 0))]
    return pairs_list_f


def _pair_op_min_id_reuse(pairs_list, used_ids):
    """Return pairs_list that has least used ids"""
    scored_pairs_list = []
    for pair in pairs_list:
        id_1 = pair[0][0]
        id_2 = pair[1][0]
        id_use_sum = used_ids[id_1] + used_ids[id_2]
        scored_pair = (pair, id_use_sum)
        scored_pairs_list.append(scored_pair)
    min_score = min([x[1] for x in scored_pairs_list])
    pairs_list_f = [x[0] for x in scored_pairs_list if x[1] == min_score]
    return pairs_list_f


def _pair_op_min_confounder(pairs_list):
    """Return pairs_list with minimal confounder difference"""
    scored_pairs_list = []
    for pair in pairs_list:
        confounder_diff = abs(pair[0][2] - pair[1][2])
        scored_pair = (pair, confounder_diff)
        scored_pairs_list.append(scored_pair)
    min_score = min([x[1] for x in scored_pairs_list])
    pairs_list_f = [x[0] for x in scored_pairs_list if x[1] == min_score]
    return pairs_list_f


def _pair_op_pick_one(pairs_list, prng):
    """Return pairs_list with minimal confounder difference"""
    pair = pairs_list[prng.choice(len(pairs_list)) - 1]
    return [pair]


def get_all_pairs(dataset):
    """Generates all possible pairs of samples.
    """
    ids_list = dataset.data[dataset.id_col].tolist()
    splits_list = dataset.data[dataset.target].tolist()
    confound_list = [None] * len(splits_list)
    if dataset.confounder:
        confound_list = dataset.data[dataset.confounder].tolist()
    samples = [(x, y, z) for x, y, z in
               zip(ids_list, splits_list, confound_list)]
    samples = sorted(samples, key=lambda x: (x[1], x[2]))
    all_pairs_list = [x for x in combinations(samples, 2)]
    return all_pairs_list


def get_sample_list(pairs_list):
    sample_list = sorted(list(set([sample[0] for pair
                                   in pairs_list
                                   for sample in pair])))
    return sample_list
