from TrainClassifiers import main

import sys

params = {"input_path"              : "/beegfs/desy/user/kasieczg/v0/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "boost"                   : True,
          "model_name"              : "test_lola_boost_0",
          "reweight_events"         : False,
          "nb_epoch"                : 100,
          "batch_size"              : 512,
          "name_train"              : "train.h5",
          "name_test"               : "test.h5",
          "name_val"                : "val.h5",
}

main(params)
