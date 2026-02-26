import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from train_isensee2017 import config
from unet3d.prediction2 import run_validation_cases


def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=None,
                         hdf5_file=config["data_file"],
                         output_label_map=True,
			 threshold = 0.1,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main()
