from subprocess import run, PIPE, STDOUT


def test_predict_synthetic_Ordinal():
    s = run(['btr-predict',
             'test_data/run_settings/settings_synthetic_Ordinal.json',
             '-g', 'test_data/hypotheses/synthetic/',
             '-o'],
            stdout=PIPE,
            stderr=STDOUT)
    assert s.returncode == 0


def test_predict_synthetic_Multiclass_Linear():
    s = run(['btr-predict',
             'test_data/run_settings/settings_synthetic_Multiclass_Linear.json',
             '-g', 'test_data/hypotheses/synthetic/',
             '-o'],
            stdout=PIPE,
            stderr=STDOUT)
    assert s.returncode == 0


def test_predict_synthetic_Multiclass_Nonlinear():
    s = run(['btr-predict',
             'test_data/run_settings/settings_synthetic_Multiclass_Nonlinear.json',
             '-g', 'test_data/hypotheses/synthetic/',
             '-o'],
            stdout=PIPE,
            stderr=STDOUT)
    assert s.returncode == 0


def test_score_synthetic_Ordinal():
    s = run(['btr-score',
             'test_data/run_settings/settings_synthetic_Ordinal.json',
             '-g', 'test_data/hypotheses/synthetic/'],
            stdout=PIPE,
            stderr=STDOUT)
    assert s.returncode == 0


def test_score_synthetic_Multiclass_Linear():
    s = run(['btr-score',
             'test_data/run_settings/settings_synthetic_Multiclass_Linear.json',
             '-g', 'test_data/hypotheses/synthetic/'],
            stdout=PIPE,
            stderr=STDOUT)
    assert s.returncode == 0


def test_score_synthetic_Multiclass_Nonlinear():
    s = run(['btr-score',
             'test_data/run_settings/settings_synthetic_Multiclass_Nonlinear.json',
             '-g', 'test_data/hypotheses/synthetic/'],
            stdout=PIPE,
            stderr=STDOUT)
    assert s.returncode == 0
