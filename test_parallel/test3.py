from mpi4py import MPI
import os
import shutil
import numpy as np
from mpipool import MPIExecutor
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
def func(x):
    return x,comm.Get_rank(),MPI.Get_processor_name()
# master process
def main():
    if rank == 0:
        if not os.path.exists("FEFFinp"):
        
            print(os.path.exists("FEFFinp"))
            os.mkdir("FEFFinp")
        else:
            print(os.path.exists("FEFFinp"))
            shutil.rmtree("FEFFinp")
            os.mkdir("FEFFinp")
        y=np.linspace(0,80,80)
        print(f"rank 0: {y}")
        #for i in range(1,size):
            #comm.send(y,dest=i,tag=i)
    #data=comm.recv(source=0,tag=rank)
    with MPIExecutor() as pool:
        pool.workers_exit()
        jobs=list(pool.map(func,y))
        for job in jobs:
            print(f"value={job[0]} at rank {job[1]} running on {job[2]}\n")
if __name__ == '__main__':
    main()


