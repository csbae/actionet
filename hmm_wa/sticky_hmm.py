from __future__ import division
import numpy as np
np.seterr(divide='ignore')
# these warnings are usually harmless for this code
#from matplotlib import pyplot as plt
import matplotlib
#import os
matplotlib.rcParams['font.size'] = 8
import pandas as pd
import pyhsmm
from pyhsmm.util.text import progprint_xrange
#import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from collections import defaultdict


def sensor_grp(s, cols):
    return filter(lambda i: any(j in i for j in s), cols)


def hmm_run(data, Nmax=6, step=100,
            kappa=12.5, alpha=3., gamma=3.):
    # Set the weak limit truncation level
    # and some hyperparameters
    obs_dim = 1
    try:
        obs_dim = data.shape[1]
    except:
        pass

    obs_hypparams = {'mu_0': np.zeros(obs_dim),
                     'sigma_0': np.eye(obs_dim),
                     'kappa_0': 10,
                     'nu_0': obs_dim+2}

    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]
    model = pyhsmm.models.WeakLimitStickyHDPHMM(
                     kappa=kappa, alpha=alpha, gamma=gamma, init_state_concentration=1.,
                     obs_distns=obs_distns)
    model.add_data(data)
    for idx in progprint_xrange(step):
        model.resample_model()

    return model


def main():
    df = pd.read_csv('../0-0/poses.csv')
    labels = pd.read_csv('../0-0/labels1.csv')
    cols = df.columns
    #selected = []
    #for c in cols[2:]:
    #    vstd, vmin, vmax = df[c].std(), df[c].min(), df[c].max()
    #    if vstd >= (vmax - vmin) / 4.0 and 'X' in c:
    #        selected.append(c)

    right_arm_cols = sensor_grp(['SBR', 'OAR', 'UAR', 'HAR', 'FIR'], cols)
    left_arm_cols = sensor_grp(['SBL', 'OAL', 'UAL', 'HAL', 'FIL'], cols)

    right_leg_cols = sensor_grp(['OSR', 'USR', 'FUR', 'FBR'], cols)
    left_leg_cols = sensor_grp(['OSL', 'USL', 'FUL', 'FBL'], cols)

    head_cols = sensor_grp(['SEH', 'KO'], cols)
    Torso_spine_cols = sensor_grp(['OHW', 'BRK', 'UHW', 'OBW', 'UBW', 'OLW', 'ULW'], cols)
    pelvis_cols = sensor_grp(['BEC'], cols)


    # data = df[right_arm_cols].values
    #selected_choose5 = np.random.choice(cols[2:], 5)
    #print(selected_choose5)
    pca = PCA(n_components=4)
    #kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True,
    #                 gamma=10)
    data0 = df[cols[2:]].values
    data = pca.fit_transform(data0)

    left_hand = labels['lefthand']
    right_hand = labels['righthand']
    trunk = labels['trunk']

    lh_dict = dict((l, i) for i, l in enumerate(left_hand.unique()))
    rh_dict = dict((r, i) for i, r in enumerate(right_hand.unique()))
    left_hand_vec = [lh_dict[k] for k in left_hand]
    right_hand_vec = [rh_dict[k] for k in right_hand]
    #Nmax = len(rh_dict)
    resultdict = defaultdict(list)
    for kappa in range(7, 9):
        for alpha in np.linspace(2.5, 4.5, 2):
            for gamma in np.linspace(2.5, 4.5, 2):
                resultdict['kappa'].append(kappa)
                resultdict['alpha'].append(alpha)
                resultdict['gamma'].append(gamma)
                model = hmm_run(data, kappa=kappa, alpha=alpha, gamma=gamma)
                labels_pred = model.stateseqs[0].tolist()
                lscore = metrics.adjusted_mutual_info_score(left_hand_vec,
                                                labels_pred)
                rscore = metrics.adjusted_mutual_info_score(right_hand_vec,
                                                 labels_pred)
                print('left hand: {:f}'.format(lscore))
                print('right hand: {:f}'.format(rscore))
                resultdict['lscore'].append(lscore)
                resultdict['rscore'].append(rscore)

    resultdf = pd.DataFrame(resultdict)
    resultdf.to_csv('hdphmm_grid_search.csv')


if __name__ == '__main__':
    main()
