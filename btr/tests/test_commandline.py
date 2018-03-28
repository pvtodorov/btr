from subprocess import run, PIPE, STDOUT
from btr.loader import Loader
from btr.utilities import get_outdir_path
import pandas as pd

settings_files = [
    'test_data/run_settings/settings_synthetic_Ordinal.json',
    'test_data/run_settings/settings_synthetic_Multiclass_Linear.json',
    'test_data/run_settings/settings_synthetic_Multiclass_Nonlinear.json']


def test_predict_synthetic():
    for s in settings_files:
        s = run(['btr-predict',
                 s,
                 '-g', 'test_data/hypotheses/synthetic/',
                 '-o'],
                stdout=PIPE,
                stderr=STDOUT)
        assert s.returncode == 0


def test_score_synthetic():
    for s in settings_files:
        s = run(['btr-score',
                 'test_data/run_settings/settings_synthetic_Ordinal.json',
                 '-g', 'test_data/hypotheses/synthetic/'],
                stdout=PIPE,
                stderr=STDOUT)
        assert s.returncode == 0


def test_score_values_synthetic():
    for s in settings_files:
        loader = Loader(settings_path=s,
                        syn_settings_overwrite=False,
                        use_synapse=False)
        folder = get_outdir_path(loader.s)
        score_csv = folder + 'score/synthetic_auc.csv'
        df = pd.read_csv(score_csv)
        df_dict = df.to_dict('records')[0]
        assert df_dict['ABCeasy_cols.txt'] >= 0
        assert df_dict['ABCeasy_cols.txt'] <= 1
        assert df_dict['ABCeasy_cols.txt'] > 0.95
        assert df_dict['ABChard_cols.txt'] >= 0
        assert df_dict['ABChard_cols.txt'] <= 1
        assert df_dict['ABChard_cols.txt'] > 0.95
        assert df_dict['BACeasy_cols.txt'] >= 0
        assert df_dict['BACeasy_cols.txt'] <= 1
        assert df_dict['BACeasy_cols.txt'] > 0.95
        assert df_dict['NS_cols.txt'] >= 0
        assert df_dict['NS_cols.txt'] <= 1
        assert df_dict['NS_cols.txt'] < df_dict['ABCeasy_cols.txt']
        assert df_dict['NS_cols.txt'] < df_dict['ABChard_cols.txt']
        assert df_dict['NS_cols.txt'] < df_dict['BACeasy_cols.txt']
        assert df_dict['NS_cols.txt'] < 0.6
