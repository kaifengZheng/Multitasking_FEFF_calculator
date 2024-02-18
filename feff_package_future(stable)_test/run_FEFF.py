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
from scipy.spatial.distance import cdist
import toml
import tomli as tomllib
import os
import argparse
import json
from tqdm import tqdm


#site_rule = config['site_rule']

######################HELP FUNCTIONS######################

def write_FEFFinp(template_dir,pos_filename,CA,site,ipot_dist,radius,numbers=0):
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
        f.write(calc_pot_atoms_list(pos_filename, absorber=CA,absorber_list=[site],ipot_dist=ipot_dist,cluster_size=radius)[0]["potential"])
        f.write('\n\n')
        f.write("ATOMS\n")
        f.write(calc_pot_atoms_list(pos_filename, absorber=CA,absorber_list=[site],ipot_dist=ipot_dist,cluster_size=radius)[0]["atoms"])
        f.write('\n')
        f.write("END\n")
    return f"FEFF_inp/{title}.inp",title
def cluster_to_numpy(clusters):
    cluster_list=[]
    ele=[]
    for cluster in clusters:
        cluster_list.append([np.float32(cluster[0]),np.float32(cluster[1]),np.float32(cluster[2])])
        ele.append(Element(cluster[4]))
    return np.array(cluster_list),ele
def neighbor_list(structure,absorber_index,ipot_dist=4.5,cluster_size=10):
    clusters=pymatgen.io.feff.inputs.Atoms(structure, int(absorber_index), cluster_size).get_lines()
    cluster_list,ele_list=cluster_to_numpy(clusters)
    neighbors=np.where(cdist([cluster_list[0]],cluster_list)[0]<ipot_dist)[0]
    neighbors=np.delete(neighbors,np.where(neighbors==0))
    neighbor_dict=dict()
    for i in range(len(neighbors)):
        key=len(np.where(cdist([cluster_list[i]],cluster_list)[0]<ipot_dist)[0])
        if key not in neighbor_dict:
            neighbor_dict[key]={ele_list[i]:[neighbors[i]]}
        else:
            if ele_list[i] not in list(neighbor_dict[key].keys()):
                neighbor_dict[key][ele_list[i]]=[neighbors[i]]
            else:
                neighbor_dict[key][ele_list[i]].append(neighbors[i])
    neighbor_rewrite=dict()

    for key in list(neighbor_dict.keys()):
        for species in list(neighbor_dict[key].keys()):
            if species not in list(neighbor_rewrite.keys()):
                neighbor_rewrite[species]=[np.array(neighbor_dict[key][species])]
            else:
                neighbor_rewrite[species].append(np.array(neighbor_dict[key][species]))
        
    return neighbor_rewrite


def calc_pot_atoms_list(path, absorber: str = None, absorber_list = [], ipot_dist = 4.5, cluster_size = 10):
    """
    Calculate the POTENTIAL and ATOMS card of feff input of given structure.
    """
    if path.split('.')[1]=='xyz':
        structure=Molecule.from_file(path)
    else:
        structure = Structure.from_file(path)
    
    pot_atoms_list = []
    
    if len(absorber_list) == 0:
        absorber_species = Element(absorber)
        absorber_list = np.where(np.array(structure.species) == absorber_species)[0]
        if absorber is None:
            raise ValueError("Please specify the absorber element.")
        
    
    for ii in range(len(absorber_list)):
        #convert structure to a cluster within a radius
        clusters=np.array(pymatgen.io.feff.inputs.Atoms(structure, int(absorber_list[ii]), cluster_size).get_lines())
        clusters = clusters[np.argsort(clusters[:, 5].astype(float))]
        #get element list and coordinates
        cluster_list,ele_list=cluster_to_numpy(clusters)
        #get neighbors of first shell atoms and catogrize them by coordination numbers
        #print(absorber_list[ii])
        neighbor_dict=neighbor_list(structure,absorber_list[ii],ipot_dist,cluster_size)
        #write POTENTIAL card for absorber
        elements=structure.elements
        num_ele_total=len(elements) #number of different elements in the cluster
        species=structure.species# list of species in the cluster
        central_element = species[absorber_list[ii]]
        ipotrow = [[0,central_element.Z,central_element.symbol,-1,-1,1,0]]
        pot_num=1 #count different potentials
        # pot_dict=dict()
        num_atom=0 #count atoms the final number should be equal to number of atoms in the cluster.
        for i in range(num_ele_total):
            num_ele_shell=0 #count the number of atoms of specific element in the cluster
            ele1=np.where(np.array(ele_list)==elements[i])[0] #index of specific element in the cluster
            #if the central atom is the same as the element of absorber, delete it from the list
            if elements[i]==central_element:
                ele1=np.delete(ele1,np.where(ele1==0))
            
            num_ele=len(np.where(np.array(ele_list)==elements[i])[0])-1#number of one kind of element
            atom_list=[]#flatten the list of catogrized atoms for this element

            #write POTENTIAL card and ATOMS card for the element
            for k in range(len(neighbor_dict[elements[i]])):
                #write POTENTIAL card
            
                ipotrow.append([pot_num,elements[i].Z,elements[i].symbol,-1,-1,len(neighbor_dict[elements[i]][k]),0])
                
                #write ATOMS card with the corresponding potential number
                for j in range(len(neighbor_dict[elements[i]][k])):
                    #elements[i]: element symbol,k: catagories of atoms(by CN), j: index of atom
                    
                    clusters[neighbor_dict[elements[i]][k][j]][3]=str(pot_num)
                    atom_list.append(neighbor_dict[elements[i]][k][j])
                # pot_dict[neighbor_dict[elements[i]][k][j]]=pot_num
                num_ele_shell+=len(neighbor_dict[elements[i]][k])
                pot_num+=1
            num_atom+=num_ele_shell
            residual=[item for item in range(len(ele1)) if item not in atom_list]
            if num_ele-num_ele_shell>0:
                ipotrow.append([pot_num,elements[i].Z,elements[i].symbol,-1,-1,num_ele-num_ele_shell,0])
                for j in range(1,len(residual)+1):
                    clusters[num_ele_shell+j][3]=str(pot_num)
                    # pot_dict[num_atom+j]=pot_num
                pot_num+=1
        unique_potential=np.unique(clusters[:,3])
        map_potential = {unique_potential[i]: str(i) for i in range(len(unique_potential))}
        pot_index=list(np.array(ipotrow)[:,0])
        #print(clusters)
        #if len(map_potential)!=len(ipotrow):
        #        miss_pot=set(pot_index).difference(list(map_potential.keys()))
        #        raise ValueError(f"The radius is too short to include all potentials, please choose a larger radius(missing potential {miss_pot}).")
        pot_atoms_list.append({"potential": tabulate(ipotrow, tablefmt="plain"), "atoms": tabulate(clusters, tablefmt="plain")})
    
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
    def __init__(self,template_dir,pos_filename,scratch,CA,ipot_dist=4.5,radius=10,site=0,numbers=0):
        self.template_dir=template_dir
        self.pos_filename=pos_filename
        self.CA=CA
        self.radius=radius
        self.scratch=scratch
        self.mode=mode
        self.cores=cores
        self.site=site
        self.numbers=numbers
        self.ipot_dist=ipot_dist
        self.title=pos_filename.split('.')[0].split('/')[1]
        self.inp_file="FEFF_inp/"+self.title+'.inp'
        self.mpi_cmd=f"mpirun -np {cores}"
        self.seq_cmd=str()

    def FEFFinp_gen(self):
        self.inp_file,self.title=write_FEFFinp(self.template_dir,self.pos_filename,self.CA,self.site,ipot_dist=self.ipot_dist,radius=self.radius,numbers=self.numbers)
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
                    FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,ipot_dist=ipot_dist,radius=radius,site=unique_index[j],numbers=numbers[j]))
                    #FEFF_obj[i].FEFFinp_gen(unique_index[j],numbers)
            else:
                try:
                    site=int(readfiles[i].split('.')[0].split('site_')[1])
                    FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,ipot_dist=ipot_dist,radius=radius,site=site,numbers=0))
                ################################################
                # if there is no site number in the file name, read site in site list
                except:
                    for j in range(len(site)):
                        FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,ipot_dist=ipot_dist,radius=radius,site=site[j],numbers=0))
                
                #exec(site_rule) #different site rule
                # #FEFF_obj.FEFFinp_gen(site)
                # ############################################### 
        start_time = time.time()
        num_obj=len(FEFF_obj)
        
        with confu.ProcessPoolExecutor(max_workers=tasks) as executor:
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
            FEFF_obj.append(FEFF_cal(template_dir,readfiles[i],scratch,CA,ipot_dist=ipot_dist,radius=radius,site=site,numbers=numbers))
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
            FEFF_obj.append(FEFF_cal(template_dir,input[i],scratch,CA,ipot_dist=ipot_dist,radius=radius,site=site,numbers=numbers))
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


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="calculation configuration")
    parser.add_argument('-w','--write_file',action='store_true',help='write FEFF input file')
    parser.add_argument('-r','--run_file',action='store_true',help='run FEFF calculation')
    args=parser.parse_args()
    config=toml.load("config.toml")
    template_dir = config['template_dir']
    pos_filename = config['pos_filename']
    scratch = config['scratch']
    CA = config['CA']
    ipot_dist = config['ipot_dist']
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
    main()


