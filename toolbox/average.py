import re
from scipy.interpolate import interp1d
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
filename=glob.glob("../output/*.json")
output=dict()
#interpolation
Energy=[]
print("reading data...\n")
for i in tqdm(range(len(filename))):
    with open(filename[i]) as f:
        particle=filename[i].split('/')[2].split("_site")[0]
        data = json.load(f)
        Energy.append(data['omega'])
        if particle not in output.keys():
            output[particle]=[{'E':np.array(data['omega'],dtype=float),
            'mu':np.array(data['mu'],dtype=float),
            'site':int(filename[i].split("_site_")[1].split('_')[0]),
            'n_sites':int(filename[i].split('n_')[1].split('.')[0])}]
            #print(particle)
        else:
            output[particle].append({'E':np.array(data['omega'],dtype=float),
            'mu':np.array(data['mu'],dtype=float),
            'site':int(filename[i].split("_site_")[1].split('_')[0]),
            'n_sites':int(filename[i].split('n_')[1].split('.')[0])})
keys=list(output.keys())
print(keys)
print("\nremeshing energy grids.../n")
energy=np.array(Energy,dtype=float)
minE,maxE=np.max(energy[:,0]),np.min(energy[:,-1])
grids=np.linspace(minE,maxE,1000)
for key in output.keys():
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
    output_ave[key]=np.round(sum1/atoms,5)




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
