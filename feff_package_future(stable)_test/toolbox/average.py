import re
from scipy.interpolate import interp1d
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import concurrent.futures as confu
from concurrent import futures
from tqdm import tqdm


parser=argparse.ArgumentParser(description="calculate average spectrum,defualt is using 1 core")
parser.add_argument('-n','--cores',type=int,default=1,help='the number of works')

args = parser.parse_args()
ncores=args.cores


def load_files(filename,i):
    particle=filename[i].split('/')[2].split("_site")[0]
    with open(filename[i]) as f:
      data = json.load(f)
    Energy=np.array(data['omega'],dtype=float)
    mu=np.array(data['mu'],dtype=float)
    site=int(filename[i].split("_site_")[1].split('_')[0])
    n_sites=int(filename[i].split('n_')[1].split('.')[0])
    return particle,Energy,mu,site,n_sites




print("reading data...\n")
output=dict()
Energy=[]
filename=glob.glob("../output/*.json")
with tqdm(total=len(filename)) as pbar:
    with confu.ThreadPoolExecutor(max_workers=ncores) as executor:
        jobs=[executor.submit(load_files,filename,i) for i in range(len(filename))]
        for job in futures.as_completed(jobs):
            Energy.append(job.result()[1])
            if job.result()[0] not in output.keys():
                output[job.result()[0]]=[{'E':job.result()[1],
                        'mu':job.result()[2],
                        'site':job.result()[3],
                        'n_sites':job.result()[4]}]
                #print(particle)
            else:
                output[job.result()[0]].append({'E':job.result()[1],
                        'mu':job.result()[2],
                        'site':job.result()[3],
                        'n_sites':job.result()[4]})
            pbar.update(1)



keys=list(output.keys())
#print(keys)
print("\nremeshing energy grids.../n")
energy=np.array(Energy,dtype=float)
minE,maxE=np.max(energy[:,0]),np.min(energy[:,-1])
grids=np.linspace(minE,maxE,300)
#grid1=np.linspace(minE,minE+2,1)
#grid2=np.linspace(minE+2,minE+35,330)
#grid3=np.linspace(minE+35,maxE,5)
#grids=np.hstack([grid1,grid2,grid3])
print(f"grids={grids}")

for key in tqdm(output.keys()):
    for i in range(len(output[key])):
        output[key][i]['mu']=interp1d(output[key][i]['E'],output[key][i]['mu'])(grids)
        output[key][i]['E']=grids

#average
print("average...\n")
output_ave=dict()
output_ave["E"]=grids
for key in tqdm(output.keys()):
    sum1=np.zeros(len(output[key][0]['E']))
    ave=np.zeros(len(output[key][0]['E']))
    atoms=0

    for i in range(len(output[key])):
        sum1+=output[key][i]['mu']*output[key][i]['n_sites']
        atoms+=output[key][i]['n_sites']
    print(f"{atoms} atoms in {key}")
    output_ave[key]=1.12*np.round(sum1/atoms,5)




print("plotting...\n")
output_ave=pd.DataFrame(output_ave)
output_ave.to_csv("../output_ave.csv",index=False)

data=pd.read_csv("../output_ave.csv")
fig,ax=plt.subplots()
for key in data.keys():
    if key!="E":
        ax.plot(data['E'],data[key],label=key)
ax.set_xlabel("E(eV)")
ax.set_ylabel("$\mu$")
plt.title("average spectra")

#plt.show()
plt.savefig("../average.png")
