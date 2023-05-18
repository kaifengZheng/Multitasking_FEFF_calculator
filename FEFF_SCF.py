import subprocess
import shutil
import glob
import datetime
import time
from pydoc import plain
import concurrent.futures as confu
from concurrent import futures
from pymatgen.core import Structure, Element,Molecule
import pymatgen.io.feff
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from tabulate import tabulate
from scipy.spatial import distance_matrix
import toml
import tomli as tomllib
import os
import argparse
import json
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from fastdist import fastdist
matplotlib.use('Agg')

parser=argparse.ArgumentParser(description="calculation configuration")
parser.add_argument('-w','--write_file',action='store_true',help='write FEFF input file')
parser.add_argument('-r','--run_file',action='store_true',help='run FEFF calculation')
args=parser.parse_args()

config=toml.load("config.toml")
template_dir = config['template_dir']
pos_filename = config['pos_filename']
scratch = config['scratch']
CA = config['CA']
radius = config['radius']
if len(config['site'])==1:
    site = config['site'][0]
else:
    site = config['site']
mode = config['mode']
cores = int(config['cores'])
tasks = int(config['tasks'])
# average = config['average']
file_type = config['file_type']
symmetry= config['symmetry']
restart=config['restart']

#SCF config
SCF_test=config['SCF_test']
particle=config['particle']
if SCF_test==True:
    rSCF=config['rSCF']
    rFMS=config['rFMS']

#site_rule = config['site_rule']

######################HELP FUNCTIONS######################

def write_FEFFinp(template_dir,pos_filename,CA,site,radius,numbers=0):
    #print(site)
    with open(template_dir) as f:
        template = f.readlines()
    #print(pos_filename)
    str_title=pos_filename.split('.')[0].split('/')[-1]
    #print(pos_filename+'\n')
    if "_site" not in str_title:
        title=pos_filename.split('.')[0].split('/')[-1]+f"_site_{site}_n_{numbers}"
    #elif "_site" not in str_title and numbers==1:
    #    title=pos_filename.split('.')[0].split('/')[-1]+f"_site_{site}"
    else:
        title=pos_filename.split('.')[0].split('/')[-1]
    with open(f"FEFF_inp//{title}.inp",'w') as f:
        f.write(f"TITLE {title}\n")
        f.writelines(template)
        f.write('\n\n')
        f.write("POTENTIALS\n")
        f.write("* ipot \t Z \t element \t l_scmt \t l_fms \t stoichiometry\n")
        f.write(calc_pot_atoms_list(pos_filename, CA,radius)[site]["potential"])
        f.write('\n\n')
        f.write("ATOMS\n")
        f.write(calc_pot_atoms_list(pos_filename, CA,radius)[site]["atoms"])
        f.write('\n')
        f.write("END\n")
    return f"FEFF_inp/{title}.inp",title

def equ_sites(path:str,absorber,cutoff,randomness=4):
    """
    :param positions: coordinates
    :param cutoff:    cutoff distance defined by mutiple scattering radius
    :return:          non-equ position indexes
    """
    # cutoff method
    def duplicates(lst, item):
        """
        :param lst: the whole list
        :param item: item which you want to find the duplication in the list
        :return: the indexes of duplicate items
        """
        return [i for i, x in enumerate(lst) if x == item]
    
    if path.split('.')[1]=='xyz':
        structure=Molecule.from_file(path)
    else:
        structure = Structure.from_file(path) 

    absorber_species = Element(absorber)
    print(absorber_species)
    absorber_list = np.where(np.array(structure.species) == absorber_species)[0]
    positions=structure.cart_coords
    dis_all =  np.around(fastdist.matrix_to_matrix_distance(np.array(positions)[absorber_list],np.array(positions), fastdist.euclidean, "euclidean"),decimals=randomness)
    dis_all.sort(axis=1)
    dis_cut = [list(dis_all[i][dis_all[i] < cutoff]) for i in range(len(dis_all))]
    dup = []
    for i in range(len(dis_cut)):
        dup.append(duplicates(dis_cut, dis_cut[i])[0])
    #unique_index = list(set(dup))  # set can delete all duplicated items in a list
    unique_index = dict()
    for i in range(len(dup)):
        if dup[i] in unique_index:
            unique_index[dup[i]].append(i)
        else:
            unique_index.update({dup[i]:[i]})
    num_sites=[]
    uni_sites=list(unique_index.keys())
    for i in range(len(uni_sites)):
        num_sites.append(len(unique_index[uni_sites[i]]))
    # sort it using sorted method. Do not use list.sort() method, because it returns a nonetype.
    #unique_index = np.array(sorted(unique_index))
    #print("number of atoms: {}".format(len(positions)))
    #print("number of unique atoms: {}".format(len(atom_index))) #
    
    return np.array(uni_sites),np.array(num_sites)  #keys are those unique sites, values are the cooresponding equ-sites for those unique_sites
def equ_sites_pointgroup(pos_dir):
    mol=Molecule.from_file(pos_dir)
    pointgroup=pymatgen.symmetry.analyzer.PointGroupAnalyzer(mol).get_equivalent_atoms()['eq_sets']
    keys=list(pointgroup.keys())
    num_sites=[len(pointgroup[keys[i]]) for i in range(len(keys))]
    return keys, num_sites
def calc_pot_atoms_list(path, absorber = None, radius = 8, absorber_list = []):
    """
    Calculate the POTENTIAL and ATOMS card of feff input of given structure.
    """
    if path.split('.')[1]=='xyz':
        structure=Molecule.from_file(path)
    else:
        structure = Structure.from_file(path)

    pot_atoms_list = []

    if len(absorber_list) == 0:
        if absorber is None:
            raise ValueError("Please specify the absorber element.")

        absorber_species = Element(absorber)
        absorber_list = np.where(np.array(structure.species) == absorber_species)[0]

    for i in absorber_list:
        pot = pymatgen.io.feff.inputs.Potential(structure, int(i))
        central_element = Element(pot.absorbing_atom)
        ipotrow = [[0, central_element.Z, central_element.symbol, -1, -1, 0.001, 0]]
        for el, amt in pot.struct.composition.items():
            ipot = pot.pot_dict[el.symbol]
            ipotrow.append([ipot, el.Z, el.symbol, -1, -1, amt, 0])

        cluster = np.array(pymatgen.io.feff.inputs.Atoms(structure, int(i), radius).get_lines())
        # sort by distance
        cluster = cluster[np.argsort(cluster[:, 5].astype(float))]

        # obtain unique potential
        unique_potential = np.unique(cluster[:, 3])
        map_potential = {unique_potential[i]: str(i) for i in range(len(unique_potential))}

        #pot index
        pot_index=list(np.array(ipotrow)[:,0])
        if len(map_potential)!=len(ipotrow):
            miss_pot=set(pot_index).difference(list(map_potential.keys()))
            raise ValueError(f"The radius is too short to include all potentials, please choose a larger radius(missing potential {miss_pot}).")
        # replace the potential label
        #cluster[:, 3] = [map_potential[str(i)] for i in cluster[:, 3]]

        # pot = []
        # for i in map_potential.keys():
        #     pot.append([map_potential[i], *ipotrow[int(i)][1:]])

        pot_atoms_list.append({"potential": tabulate(ipotrow, tablefmt="plain"), "atoms": tabulate(cluster, tablefmt="plain")})

    return pot_atoms_list

def run_mpi(cores,run_dir):
    subprocess.run("cd "+ os.path.dirname(f"{run_dir}")+ f"&&feffmpi {cores}>>feff.out", shell=True)
def run_seq(run_dir):
    subprocess.run("cd "+ os.path.dirname(f"{run_dir}")+ f"&&feff >>feff.out", shell=True)
def write_files(js,inp_filename):
    print(inp_filename)
    title=inp_filename.split('.')[0].split('/')[-1]
    if not os.path.exists('output'):

        os.mkdir('output')
    json.dump(js, open(f"output/{title}.json", 'w'))
def FEFF_obj_fun(obj,i):
    return obj[i].particle_run()
def run_write(obj):
    return obj.FEFFinp_gen()
def write_outlog(content):
    with open("output.dat","a") as file1:
        file1.write(content+'\n')

def write_templete_SCF(rfms=7,rscf=6):
    templete_array=['EDGE L3\n',
                     'S02  0.9\n',
                     'COREHOLE NONE\n',
                     'CONTROL 1 1 1 1 1 1\n',
                     'XANES 8 0.05 0.1\n',
                     f'FMS  {rfms}\n',
                     'EXCHANGE 0 0 0 -1\n',
                     'DEBYE 80 230 0\n',
                     f'SCF {rscf} 0 100 0.1 15\n']

    with open("template.inp",'w') as f:
        f.writelines(templete_array)

class FEFF_cal:
    def __init__(self,template_dir,pos_filename,scratch,CA,radius,site=0,numbers=0):
        self.template_dir=template_dir
        self.pos_filename=pos_filename
        self.CA=CA
        self.radius=radius
        self.scratch=scratch
        self.mode=mode
        self.cores=cores
        self.site=site
        self.numbers=numbers
        #print(pos_filename)
        self.title=pos_filename.split('.')[0].split('/')[1]
        self.inp_file="FEFF_inp/"+self.title+'.inp'
        self.mpi_cmd=f"mpirun -np {cores}"
        self.seq_cmd=str()

    def FEFFinp_gen(self):
        self.inp_file,self.title=write_FEFFinp(self.template_dir,self.pos_filename,self.CA,self.site,self.radius,self.numbers)
        #print(f"writing {self.title}")
    def particle_run(self):

        run_dir = f"{self.scratch}/{config['name']}/{self.title}"
        #subprocess.run(f"echo running FEFF on {self.title}...>> output.log",shell=True)
        #write_outlog(f"running FEFF on {self.title}...")
        if os.path.exists(f"{run_dir}"):
            shutil.rmtree(run_dir)
        if not os.path.exists(f"{run_dir}"):
            os.makedirs(f"{run_dir}")
        
        shutil.copyfile(self.inp_file, f"{run_dir}/feff.inp")
        if self.mode=='mpi_seq' or self.mode=='mpi_multi':
            start_time = time.time()  
            with open(f"{run_dir}/feff.out",'w') as f1:
                f1.write(f"----RUNNING FEFF ON {self.title} with {cores} cores-----\n\n")
                
            a=subprocess.run([f"cd {run_dir} && feffmpi {cores} >>feff.out","wait"],shell=True) #cd {run_dir} &&pwdfeffmpi {cores}>>feff.out
            #subprocess.run(f"echo {a.args[0]}>>output.log",shell=True)
            #subprocess.run(f"returncode={str(a.returncode[0])}>>output.log",shell=True)
            write_outlog(f"{a}")
            finish_time = time.time()
        if self.mode=='seq_seq' or self.mode=='seq_multi':
            start_time = time.time()  
            with open(f"{run_dir}/feff.out",'w') as f1:
                f1.write(f"----RUNNING FEFF ON {self.title} with {cores} cores-----\n\n")
            a=subprocess.run([f"cd {run_dir} && feff >>feff.out","wait"], shell=True)
            #subprocess.run(f"echo {a.args[0]}>>output.log",shell=True)
            #subprocess.run(f"echo returncode={str(a.returncode[0])}",shell=True)
            #print(a)
            write_outlog(f"{a}")
            finish_time = time.time()
        with open(f'{run_dir}/feff.inp') as f:
            feffinp = f.readlines()
        with open(f'{run_dir}/feff.out') as f:
            feffout = f.readlines()
        if os.path.exists(f'{run_dir}/xmu.dat'):
            with open(f'{run_dir}/xmu.dat') as f:
                xmudat = f.readlines()

            endhindex=xmudat.index('#  omega    e    k    mu    mu0     chi     @#\n')
            js = {
                'fname': self.title,
                'header': xmudat[0:endhindex+1],
                'omega': [str(np.float32(i.split()[0])) for i in xmudat[endhindex+1:]],
                'e':     [str(np.float32(i.split()[1])) for i in xmudat[endhindex+1:]],
                'mu':    [str(np.float32(i.split()[3])) for i in xmudat[endhindex+1:]],
                'mu0':   [str(np.float32(i.split()[4])) for i in xmudat[endhindex+1:]],
                'chi':   [str(np.float32(i.split()[5])) for i in xmudat[endhindex+1:]],
                'feffout': feffout,
                'time': finish_time-start_time,
                'ncores': str(np.int8(self.cores)),
                } 
        else:
            with open(f"{run_dir}/feff.out","w+") as f1:
                f1.write("FEFF calculation failed!\n\n")
            js = {
                'time_elapsed': finish_time - start_time,
                'feffout': feffout,
                }
        shutil.rmtree(run_dir) #test comment this line
        return js,self.inp_file
    def input_generator(unique_index):
        for i in range(len(unique_index)):
            yield FEFF_cal(template_dir,filelist[i],scratch,CA,radius,site=config['site'][i],numbers=numbers[i])
def writing_process():
    readfiles=glob.glob(f"input/{file_type}")
    if type(readfiles)==str:
        readfiles=[readfiles]
    FEFF_obj=[]

    if not os.path.exists("FEFF_inp"):
        os.mkdir("FEFF_inp")
    else:
        shutil.rmtree("FEFF_inp")
        os.mkdir("FEFF_inp")


    for i in range(len(readfiles)):
        if particle=='atom':
            if len(config['site'])==1:
                iter=FEFF_cal(template_dir,readfiles[i],scratch,CA,radius,site=config['site'][0],numbers=1)
                run_write(iter)
            else:
                for i in tqdm(range(len(config['site'])),total=len(config['site'])):
                    iter=FEFF_cal(template_dir,readfiles[i],scratch,CA,radius,site=config['site'][i],numbers=1)
                    run_write(iter)
        if particle=='particle':
                unique_index,numbers = equ_sites_pointgroup(readfiles[i])
                for j in range(len(unique_index)):
                    FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,radius,site=unique_index[j],numbers=numbers[j]))
                    #FEFF_obj[i].FEFFinp_gen(unique_index[j],numbers)
                    start_time = time.time()
                    num_obj=len(FEFF_obj)
                with confu.ProcessPoolExecutor(max_workers=tasks) as executor:
                    jobs=list(tqdm(executor.map(run_write,FEFF_obj),total=num_obj))
                finish_time = time.time()

def run_process_from_fresh():
    
        readfiles=glob.glob(f"FEFF_inp/*.inp")
        if type(readfiles)==str:
            readfiles=[readfiles]
        FEFF_obj=[]
        for i in tqdm(range(len(readfiles))):
            #print(readfiles[i])
            site=int(readfiles[i].split('.')[0].split('site_')[1].split('_n')[0])
            numbers=int(readfiles[i].split('.')[0].split('n_')[1])
            #print(numbers)
            FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,radius,site=site,numbers=numbers))
        if mode=='seq_multi':
            start_time = time.time() 
            with confu.ProcessPoolExecutor(max_workers=tasks) as executor:
                #jobs=list(executor.map(FEFF_obj_fun,FEFF_obj))
                #for job in jobs:
                #    print(job)
                #    write_files(job[1],job[0])
                jobs=[executor.submit(FEFF_obj_fun,FEFF_obj,i) for i in range(len(FEFF_obj))]
                for job in futures.as_completed(jobs):
                    write_files(job.result()[0],job.result()[1])
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)
        if mode=='seq_seq':
            start_time = time.time() 
            for i in range(len(readfiles)):
                js,inp_file=FEFF_obj[i].particle_run()
                write_files(js,inp_file)
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)
        if mode=='mpi_seq':
            start_time = time.time() 
            for i in range(len(readfiles)):
                js,inp_file=FEFF_obj[i].particle_run()
                write_files(js,inp_file)
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)
        if mode=='mpi_multi':
            start_time = time.time() 
            with confu.ProcessPoolExecutor(max_workers=tasks) as executor:
                #jobs=list(executor.map(FEFF_obj_fun,FEFF_obj))
                #for job in jobs:
                #    write_files(job[1],job[0])
                jobs=[executor.submit(FEFF_obj_fun,FEFF_obj,i) for i in range(len(FEFF_obj))]
                for job in futures.as_completed(jobs):
                    write_files(job.result()[0],job.result()[1])
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)
    
def run_process_from_restart():
    readfiles=glob.glob(f"FEFF_inp/*.inp")
    readout=glob.glob(f"output/*.json")
    out = []
    input = []
    for str1 in readout:
        out.append(str1.split('/')[1].split('.')[0])
    for i in range(len(readfiles)):
        if readfiles[i].split('/')[1].split('.')[0] in out:
            continue
        else:
            input.append(readfiles[i])
        
        if type(input)==str:
            input=[input]
        FEFF_obj=[]
        

        for i in tqdm(range(len(input))):
            site=int(input[i].split('.')[0].split('site_')[1].split('_n')[0])
            numbers=int(input[i].split('.')[0].split('n_')[1])
            #print(numbers)
            FEFF_obj.append(FEFF_cal(template_dir,input[i],scratch,CA,radius,site=site,numbers=numbers))
        if mode=='seq_multi':
            start_time = time.time() 
            with confu.ProcessPoolExecutor(max_workers=tasks) as executor:
                #jobs=list(executor.map(FEFF_obj_fun,FEFF_obj))
                #for job in jobs:
                #    print(job)
                #    write_files(job[1],job[0])
                jobs=[executor.submit(FEFF_obj_fun,FEFF_obj,i) for i in range(len(FEFF_obj))]
                for job in futures.as_completed(jobs):
                    write_files(job.result()[1],job.result()[0])
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)
        if mode=='seq_seq':
            start_time = time.time() 
            for i in range(len(readfiles)):
                js,inp_file=FEFF_obj[i].particle_run()
                write_files(js,inp_file)
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)
        if mode=='mpi_seq':
            start_time = time.time() 
            for i in range(len(readfiles)):
                js,inp_file=FEFF_obj[i].particle_run()
                write_files(js,inp_file)
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)
        if mode=='mpi_multi':
            start_time = time.time() 
            with confu.ProcessPoolExecutor(max_workers=tasks) as executor:
                #jobs=list(executor.map(FEFF_obj_fun,FEFF_obj))
                #for job in jobs:
                #    write_files(job[1],job[0])
                jobs=[executor.submit(FEFF_obj_fun,FEFF_obj,i) for i in range(len(FEFF_obj))]
                for job in futures.as_completed(jobs):
                    write_files(job.result()[1],job.result()[0])
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)

def run_check():
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
    readout=glob(f"output/*.json")
    if type(readout)==str:
        readout=[readout]
    delete=0
    with tqdm(total=len(readout)) as pbar:
        with confu.ProcessPoolExecutor(max_workers=cores) as executor:
            jobs=[executor.submit(check_files,readout,i) for i in range(len(readout))]
            for job in futures.as_completed(jobs):
                pbar.update(1)

def test_results(filename):
    with open(filename,'r') as f1:
        data=json.load(f1)
    Energy=np.array(data['omega'],dtype=float)
    mu=np.array(data['mu'],dtype=float)
    return Energy, mu
def mu_regrid_p(filename):
    E_p=[]
    mu_p=[]
    E_pf=[]
    E_pl=[]
    mu_interp=[]
    for i in range(len(filename)):
        E_p.append(test_results(filename[i])[0])
        E_pf.append(test_results(filename[i])[0][0])
        E_pl.append(test_results(filename[i])[0][-1])
        mu_p.append(test_results(filename[i])[1])
    E_min=np.max(np.array(E_pf))
    E_max=np.min(np.array(E_pl))
    new_E=np.linspace(E_min,E_max,1000)
    for i in range(len(filename)):
        mu_interp.append(interp1d(E_p[i],mu_p[i])(new_E))
    return new_E, mu_interp
def mu_regrid_E_mu(E,mu):
    E_p=[]
    mu_p=[]
    E_pf=[]
    E_pl=[]
    mu_interp=[]
    for i in range(len(E)):
        E_p.append(E[i])
        E_pf.append(E[i][0])
        E_pl.append(E[i][-1])
        mu_p.append(mu[i])

        E_min=np.max(np.array(E_pf))
        E_max=np.min(np.array(E_pl))
        new_E=np.linspace(E_min,E_max,1000)
    for i in range(len(E)):
        mu_interp.append(interp1d(E_p[i],mu_p[i])(new_E))

    return new_E, mu_interp
def average_mu(mu_regrid):
    mu_regrid_ave=np.array(mu_regrid).mean(axis=0)
    return mu_regrid_ave
def SCF_test_run():
    rSCF=config['rSCF']
    rFMS=config['rFMS']
    configuration =[[a,b] for a in rSCF 
                            for b in rFMS]
    E_iter=[]
    mu_iter=[]
    con_save=[]
    for example, con in tqdm(enumerate(configuration),total=len(configuration)):
        write_templete_SCF(rfms=con[1],rscf=con[0])
        writing_process()
        #try:
        run_process_from_fresh()
        #except Exception as e:
        #    subprocess.run(f'echo {e} >> output.log',shell=True)
        #    exit()
       # if args.run_file==True and restart==True:
       #     run_check()
       #     try:
       #         run_process_from_restart()
       #     except Exception as e:
       #         subprocess.run(f'echo {e} >> output.log',shell=True)
       #         exit()
        filename=glob.glob(f"output/*.json")
           
        if particle=='atom':
             E_iter.append(test_results(filename[0])[0])
             mu_iter.append(test_results(filename[0])[1])
        if particle=='particle':
             E_iter=mu_regrid_p(filename)[0]
             mu=mu_regrid_p(filename)[1]
             mu_ave=average_mu(mu)
             mu_iter.append(mu_ave)
        con_save.append(con)
        #print(mu_iter,E_iter)

    E_inter1,mu_inter1=mu_regrid_E_mu(E_iter,mu_iter)
    E_inter1=np.array(E_inter1)
    mu_inter1=np.array(mu_inter1)
    error_data={'rSCF':[],'rFMS':[],'error':[0]*len(con_save),'mu':[]}
    #print(len(mu_inter1[0]))
    #error_data['error']=[np.sum(mu_inter1[1]-mu_inter1[0])]
    
    for i in range(len(con_save)):
        error_data['rSCF'].append(con_save[i][0])
        error_data['rFMS'].append(con_save[i][1])
        error_data['mu'].append(mu_inter1[i])
    
    error_table=pd.DataFrame(error_data)

    rSCF_num=np.unique(np.array(error_table['rSCF']))
    
    for i in range(len(rSCF_num)):
        SCFtable=error_table[error_table['rSCF']==rSCF_num[i]]
        SCFtable.sort_values(by=['rFMS'])
        #print(error_table[(error_table['rSCF']==rSCF_num[i])&(error_table['rFMS']==np.min(np.array(rFMS)))]['error'])
        error_table.loc[(error_table['rSCF']==rSCF_num[i])&(error_table['rFMS']==np.min(np.array(rFMS))),'error']=np.abs(np.sum(np.array(SCFtable.iloc[1]['mu'])-np.array(SCFtable.iloc[0]['mu'])))
        for j in range(0,len(SCFtable)-1):
            error_table.loc[(error_table['rSCF']==rSCF_num[i])&(error_table['rFMS']==SCFtable.iloc[j+1]['rFMS']),'error']=np.abs(np.sum(np.array(SCFtable.iloc[j+1]['mu'])-np.array(SCFtable.iloc[j]['mu'])))


            
   
   
   
   
    np.save('E.npy',E_inter1)
    error_table.to_csv('error_table.csv')

    fig,ax=plt.subplots()
    for i in range(len(rSCF_num)):
        ax.plot(np.array(error_table[error_table['rSCF']==rSCF_num[i]]['rFMS']),
                np.array(error_table[error_table['rSCF']==rSCF_num[i]]['error']),
                marker=11,linestyle='solid',color=f"C{i}",label=f"rSCF={rSCF_num[i]}")
    ax.set_xlabel('rFMS')
    ax.set_ylabel('error')
    #ax.set_ylim([0,5])
    ax.legend(frameon=False)
    fig.savefig('error.png')
    fig,ax=plt.subplots()
    for i in range(len(con_save)):
        ax.plot(E_inter1,np.array(error_table.iloc[i]['mu']),label=f"rSCF={error_table.iloc[i]['rSCF']} rFMS={error_table.iloc[i]['rFMS']}")
    ax.legend(frameon=False)
    fig.savefig('spectra.png')
def main():
    comm=MPI.COMM_WORLD
    name=MPI.Get_Processor_name()
    if config['SCF_test']==True:
        SCF_test_run()
    if config['SCF_test']==False:
        if args.write_file==True:
            writing_process()
        if args.run_file==True and restart==False:
            try:
                run_process_from_fresh()
            except Exception as e:
                subprocess.run(f'echo {e} >> output.log',shell=True)
                exit()
        if args.run_file==True and restart==True:
            run_check()
            try:
                run_process_from_restart()
            except Exception as e:
                subprocess.run(f'echo {e} >> output.log',shell=True)
                exit()
if __name__ == '__main__':
    main()
    
