import os

from Uformer_Info.dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestSR
# opt.train_dir,   img_options：{'patch_size': opt.train_ps} (train_patchsize 默认 128)
# 加载图片时 已经除了255
# [C, W, H]
def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)  # clean, noisy, clean_filename, noisy_filename

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


def get_test_data_SR(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestSR(rgb_dir, None)