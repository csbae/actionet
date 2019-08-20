import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import animation
from matplotlib.colors import PowerNorm
from ihdphmm import iHDPHMM
import pandas as pd
# import ipdb

if __name__ == '__main__':    
    np.random.seed(11)
    H = 3
    L = 30
    colors = ['r', 'b', 'g']

    # Read in the data and group them by 'doc_id'
    datafile = u'../data/mit_trj_parkinglot_all_hilbert100.csv'
    df = pd.read_csv(datafile)
    data = []
    dfgs = df.groupby('doc_id')

    # Construct lists with Hilbert indices
    for k in dfgs.groups.keys():
        tmp = df.ix[dfgs.groups[k]]
        data.append(tmp['hilbert_idx'].values)

    # Initialize the iHDP-HMM model and sample it
    model = iHDPHMM(data)
    for i in range(2):
        model.sampler()

    # Print out the state and PI of the model
    print(model.state)
    print(model.PI)

