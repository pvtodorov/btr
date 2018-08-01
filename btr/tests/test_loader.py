from btr.loader import Loader, get_or_create_syn_folder
from btr.utilities import get_settings_md5
from synapseclient import File

settings_md5_list = [('settings_synthetic_Ordinal.json',
                      'f84c49724196da1ed7a7f4dcc0f0877b')]


def test_load_settings():
    # load each settings file in the list
    # check that the md5 we expect is the md5 we get
    for sp in settings_md5_list:
        loader = Loader(settings_path='test_data/run_settings/' + sp[0],
                        syn_settings_overwrite=False)
        md5 = get_settings_md5(loader.settings)
        assert(sp[1] == md5)


def test_synapse_settings_match_synapse_md5s():
    # load each settings file in the list
    # check that the md5 we expect is the md5 we get
    for sp in settings_md5_list:
        loader = Loader(settings_path='test_data/run_settings/' + sp[0],
                        syn_settings_overwrite=False)
        loader = Loader('test_data/run_settings/' + sp[0])
        md5 = get_settings_md5(loader.settings)
        parent = get_or_create_syn_folder(loader.syn,
                                          'run_settings/',
                                          'syn11974673', create=False)
        file = File('test_data/run_settings/' + sp[0], parent=parent)
        remote_file = loader.syn.get(file, downloadFile=False)
        remote_md5 = remote_file.annotations.get('settings_md5')
        md5 = get_settings_md5(loader.settings)
        assert([md5] == remote_md5)
