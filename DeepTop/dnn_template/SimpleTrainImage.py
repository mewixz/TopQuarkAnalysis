
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from TrainClassifiers import main

import sys

params = {"input_path"              : "/beegfs/desy/user/kasieczg/TopImages_RePre/",
          "output_path"             : "./",
          "inputs"                  : "2d",
          "model_name"              : sys.argv[1],
          "reweight_events"         : 0,
          "early_stop_patience"     : 1,
          "nb_epoch"                : 30,
          "batch_size"              : 512,
          "name_train"              : "train_img-resort.h5",
          "name_test"               : "test_img-resort.h5",
          "name_val"                : "val_img-resort.h5",
}

main(params)
