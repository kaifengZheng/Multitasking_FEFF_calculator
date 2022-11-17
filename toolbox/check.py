from glob import glob
import concurrent.futures as confu
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
from concyrrent import futures

parser=argparse.ArgumentParser(description="check corrupted files")
parser.add_argument('-n','--cores',type=int,default=1,help='the number of works')
args = parser.parse_args()
ncores=args.cores

def check_files(filename,i):
    particle=filename[i].split('/')[2].split("_site")[0]
    try:
        with open(filename[i]) as f:
            data = json.load(f)
        Energy=np.array(data['omega'],dtype=float)
        mu=np.array(data['mu'],dtype=float)
        site=int(filename[i].split("_site_")[1].split('_')[0])
        n_sites=int(filename[i].split('n_')[1].split('.')[0])
    except:
        os.remove(filename[i])
        delete+=1
        print(f"""{readout[i]} is corrupted and deleted\n""")
        print(f"deleted {delete} corrupted files\n")


readout=glob.glob(f"../output/*.json")
if type(readout)==str:
    readout=[readout]
delete=0
with tqdm(total=len(readout)) as pbar:
    with confu.ThreadPoolExecutor(max_workers=ncores) as executor:
        jobs=[executor.submit(check_files,readout,i) for i in range(len(readout))]
        for job in futures.as_completed(jobs):
            pbar.update(1)