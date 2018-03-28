from subprocess import run, PIPE, STDOUT

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


