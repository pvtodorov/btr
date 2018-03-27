from btr.loader import Loader, get_or_create_syn_folder
from btr.utilities import get_settings_md5
from synapseclient import File

settings_md5_list = [('settings_LPOCV_BM36_Multiclass_Linear.json',
                      'c357e1d9319d1deac8d33fcc6a8aad16'),
                     ('settings_LPOCV_BM36_Multiclass_Linear_BC.json',
                      '2e8af8de6fe6d84e6ba37b882353c0fe'),
                     ('settings_LPOCV_BM36_Multiclass_Nonlinear.json',
                      'aa8913c242110beb633f78b1298e6ee8'),
                     ('settings_LPOCV_BM36_Multiclass_Linear_AC.json',
                      'a6575c9fc9aeb0141ff7fbdf3a6b9f1d'),
                     ('settings_LPOCV_BM36_Multiclass_Linear_AB.json',
                      '61b6aa3dd90b6fb9446df6a6de9c79d4'),
                     ('settings_ROSMAP_LPOCV_BM9_46_Ordinal.json',
                      'a53e7112a550ee5ce84b4b9b1bb19266'),
                     ('settings_LPOCV_BM22_Multiclass_Linear_AB.json',
                      '5ecd17d3fc8e201c2d17424d133e6b12'),
                     ('settings_LPOCV_BM22_Multiclass_Linear.json',
                      'e587091987f44077e8285de27fd2651e'),
                     ('settings_ROSMAP_LPOCV_BM9_46_Multiclass_Linear_AB.json',
                      '11ca003d2b62278f70791ab53e0ac9f2'),
                     ('settings_ROSMAP_LPOCV_BM9_46_Multiclass_Linear_AC.json',
                      '46943d45ef37c2d24b27a2326e9eb7a1'),
                     ('settings_LPOCV_BM22_Multiclass_Linear_AC.json',
                      'b801a9179b4ea353bcd8c363ad933b99'),
                     ('settings_LPOCV_BM22_Multiclass_Nonlinear.json',
                      '9e1e648693c5dccdd8ea11a18d749213'),
                     ('settings_LPOCV_BM22_Ordinal.json',
                      '575e42610b0b59afa9dff673f4c99967'),
                     ('settings_ROSMAP_LPOCV_BM9_46_Multiclass_Linear_BC.json',
                      '6c9b1972eeec65022eda0c01db410c77'),
                     ('settings_LPOCV_BM22_Multiclass_Linear_BC.json',
                      '40d8e36d2477e7959822f711efecaab7'),
                     ('settings_LPOCV_BM36_Ordinal.json',
                      '2534b5f6e938fe3127681da9d5ecd43b')]


def test_load_settings():
    # load each settings file in the list
    # check that the md5 we expect is the md5 we get
    for sp in settings_md5_list:
        loader = Loader(settings_path='test_data/run_settings/' + sp[0],
                        syn_settings_overwrite=False)
        md5 = get_settings_md5(loader.s)
        assert(sp[1] == md5)


def test_synapse_settings_match_synapse_md5s():
    # load each settings file in the list
    # check that the md5 we expect is the md5 we get
    for sp in settings_md5_list:
        loader = Loader(settings_path='test_data/run_settings/' + sp[0],
                        syn_settings_overwrite=False)
        loader = Loader('test_data/run_settings/' + sp[0])
        md5 = get_settings_md5(loader.s)
        parent = get_or_create_syn_folder(loader._syn, 'run_settings/',
                                          'syn11615746', create=False)
        file = File('test_data/run_settings/' + sp[0], parent=parent)
        remote_file = loader._syn.get(file, downloadFile=False)
        remote_md5 = remote_file.annotations.get('settings_md5')
        md5 = get_settings_md5(loader.s)
        assert([md5] == remote_md5)
