import json
import os
import pandas as pd


def main():
    datafile = 'mit_trj_parkinglot_all.json'
    datafolder = u'/home/sjj/workspace/iHDP_HMM/data'
    fulldatafile = os.path.join(datafolder, datafile)
    target_fld = '/home/sjj/workspace/iHDP_HMM/data/mit_trj'
    
    with open(fulldatafile) as fh:
        data = json.load(fh)
    
    trajs = data['trajectories']['trajectory']
    for idx, traj in enumerate(trajs):
        print('processing {:d}'.format(idx))
        points = traj['points']
        df = pd.DataFrame(points)
        df.to_csv(os.path.join(target_fld, 'md_{:d}'.format(idx)), index=False)


if __name__ == '__main__':
    main()


