import pickle

import matplotlib as mpl
mpl.use('Agg')
import pandas
import numpy as np
import matplotlib.pyplot as plt

input_filename = "/beegfs/desy/user/kasieczg/v0/train.h5"
store = pandas.HDFStore(input_filename)

# Read the first 10 events
foo = store.select("table",stop=100000)

Es = ["E_{0}".format(i) for i in range(40)]
PXs = ["PX_{0}".format(i) for i in range(40)]
PYs = ["PY_{0}".format(i) for i in range(40)]
PZs = ["PZ_{0}".format(i) for i in range(40)]

foo["E_tot"] = foo[Es].sum(axis=1)
foo["PX_tot"] = foo[PXs].sum(axis=1)
foo["PY_tot"] = foo[PYs].sum(axis=1)
foo["PZ_tot"] = foo[PZs].sum(axis=1)

foo["M"] = np.sqrt(pow(foo["E_tot"],2) - pow(foo["PX_tot"],2) - pow(foo["PY_tot"],2) - pow(foo["PZ_tot"],2))


bins = np.arange(10,280,5)

counts_qcd, bins_qcd, bars = plt.hist(foo[foo["is_signal_new"]==0]["M"],bins=bins,alpha=0.5,label="QCD")
counts_top, bins_top, bars = plt.hist(foo[foo["is_signal_new"]==1]["M"],bins=bins,alpha=0.5,label="top")

di = {}
di["counts_qcd"] = counts_qcd
di["bins_qcd"] = bins_qcd
di["counts_top"] = counts_top
di["bins_top"] = bins_top

f = open("mass_weights.pickle", "wb")
pickle.dump(di, f)

plt.xlabel('Mass [GeV]')
plt.ylabel('Fraction of Jets')
plt.legend()

plt.savefig("foo.png")
