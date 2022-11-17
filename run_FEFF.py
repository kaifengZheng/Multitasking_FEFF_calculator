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
average = config['average']
file_type = config['file_type']
symmetry= config['symmetry']
restart=config['restart']
#site_rule = config['site_rule']

######################HELP FUNCTIONS######################

def write_FEFFinp(template_dir,pos_filename,CA,site,radius,numbers=0):
    with open(template_dir) as f:
        template = f.readlines()
    str_title=pos_filename.split('.')[0].split('/')[-1]
    if "_site" not in str_title and numbers!=0:
        title=pos_filename.split('.')[0].split('/')[-1]+f"_site_{site}_n_{numbers}"
    elif "_site" not in str_title and numbers==0:
        title=pos_filename.split('.')[0].split('/')[-1]+f"_site_{site}"
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

def equ_sites(CA:str,labels,natoms,positions,cutoff,randomness=4):
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
    
    caatoms=[i for i, e in enumerate(labels) if e == CA]

    dis_all =  np.around(distance_matrix(np.array(positions),np.array(positions)[caatoms]),decimals=randomness)
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
    # formula_text
    formula = '$'
    for i,(k,v) in enumerate(natoms.items()):
        formula = formula + k + '_{' + str(int(v)) + '}'
    formula = formula + '$'
    # sort it using sorted method. Do not use list.sort() method, because it returns a nonetype.
    #unique_index = np.array(sorted(unique_index))
    #print("number of atoms: {}".format(len(positions)))
    #print("number of unique atoms: {}".format(len(atom_index))) #
    
    return unique_index,formula  #keys are those unique sites, values are the cooresponding equ-sites for those unique_sites
def equ_sites_pointgroup(pos_dir):
    mol=Molecule.from_file(pos_dir)
    pointgroup=pymatgen.symmetry.analyzer.PointGroupAnalyzer(mol).get_equivalent_atoms()['eq_sets']
    keys=list(pointgroup.keys())
    num_sites=[len(pointgroup[keys[i]]) for i in range(len(keys))]
    return keys, num_sites
    # running process
def run_mpi(cores,run_dir):
    subprocess.run("cd "+ os.path.dirname(f"{run_dir}")+ f"&&feffmpi {cores}>>feff.out", shell=True)
def run_seq(run_dir):
    subprocess.run("cd "+ os.path.dirname(f"{run_dir}")+ f"&&feff >>feff.out", shell=True)
def write_files(inp_filename,js):
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
        self.title=pos_filename.split('.')[0].split('/')[1]
        self.inp_file="FEFF_inp/"+self.title+'.inp'
        self.mpi_cmd=f"mpirun -np {cores}"
        self.seq_cmd=str()

    def FEFFinp_gen(self):
        self.inp_file,self.title=write_FEFFinp(self.template_dir,self.pos_filename,self.CA,self.site,self.radius,self.numbers)
        #print(f"writing {self.title}")
    def particle_run(self):

        run_dir = f"{self.scratch}/{self.title}"
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
    




def main():

    if args.write_file==True:
        readfiles=glob.glob(f"input/{file_type}")
        if type(readfiles)==str:
           readfiles=[readfiles]
        FEFF_obj=[]
           
        if not os.path.exists("FEFF_inp"):
           os.mkdir("FEFF_inp")
        else:
           shutil.rmtree("FEFF_inp")
           os.mkdir("FEFF_inp")

        for i in tqdm(range(len(readfiles))):
            if symmetry:
                unique_index,numbers = equ_sites_pointgroup(readfiles[i])
                for j in range(len(unique_index)):
                    FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,radius,site=unique_index[j],numbers=numbers[j]))
                    #FEFF_obj[i].FEFFinp_gen(unique_index[j],numbers)
            else:
                ################################################
                site=int(readfiles[i].split('.')[0].split('site_')[1])
                FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,radius,site=site,numbers=0))
                #exec(site_rule) #different site rule
                # #FEFF_obj.FEFFinp_gen(site)
                # ############################################### 
        start_time = time.time()
        num_obj=len(FEFF_obj)
        with confu.ThreadPoolExecutor(max_workers=tasks) as executor:
            jobs=list(tqdm(executor.map(run_write,FEFF_obj),total=num_obj))
            finish_time = time.time() 
             
             
             
    if args.run_file==True and restart==False:
        readfiles=glob.glob(f"FEFF_inp/*.inp")
        if type(readfiles)==str:
            readfiles=[readfiles]
        FEFF_obj=[]
        

        for i in tqdm(range(len(readfiles))):
            site=int(readfiles[i].split('.')[0].split('site_')[1].split('_n')[0])
            numbers=int(readfiles[i].split('.')[0].split('n_')[1])
            #print(numbers)
            FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,radius,site=site,numbers=numbers))
        if mode=='seq_multi':
            start_time = time.time() 
            with confu.ThreadPoolExecutor(max_workers=tasks) as executor:
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
            with confu.ThreadPoolExecutor(max_workers=tasks) as executor:
                #jobs=list(executor.map(FEFF_obj_fun,FEFF_obj))
                #for job in jobs:
                #    write_files(job[1],job[0])
                jobs=[executor.submit(FEFF_obj_fun,FEFF_obj,i) for i in range(len(FEFF_obj))]
                for job in futures.as_completed(jobs):
                    write_files(job.result()[1],job.result()[0])
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)

        readout=glob.glob(f"output/*.json")
        delete=0
        for i in range(len(readout)):
            try:    
                with open(readout[i]) as f:
                    data = json.load(f)
                Energy=np.array(data['omega'],dtype=float)
                mu=np.array(data['mu'],dtype=float)
            except:
                os.remove(readout[i])
                write_outlog(f"{readout[i]} is removed!")
                delete+=1
        readout2=glob.glob(f"output/*.json")
        write_outlog(f"delete {delete} files...")
        write_outlog(f"have {len(readfiles)} files...")
        write_outlog(f"calculate {len(readout2)}...")




    if args.run_file==True and restart==True:
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
            with confu.ThreadPoolExecutor(max_workers=tasks) as executor:
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
            with confu.ThreadPoolExecutor(max_workers=tasks) as executor:
                #jobs=list(executor.map(FEFF_obj_fun,FEFF_obj))
                #for job in jobs:
                #    write_files(job[1],job[0])
                jobs=[executor.submit(FEFF_obj_fun,FEFF_obj,i) for i in range(len(FEFF_obj))]
                for job in futures.as_completed(jobs):
                    write_files(job.result()[1],job.result()[0])
            finish_time = time.time()
            subprocess.run(f"echo End in {(finish_time-start_time)/60} min >>output.log",shell=True)


if __name__ == '__main__':
    main()


