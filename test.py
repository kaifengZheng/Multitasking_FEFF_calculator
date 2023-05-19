import numpy as np
#from mpi4py.futures import MPIPoolExecutor,MPICommExecutor
import concurrent.futures as confu
from concurrent import futures
from mpipool import MPIExecutor
from mpipool import MPIPool
from mpi4py import MPI
import time
from itertools.chain import from_iterable
def func(x):
    return x,MPI.COMM_WORLD.Get_rank(),MPI.Get_processor_name()
def gen(y):
    for i in range(len(y)):
        yield func(y[i])
def main():
    y=np.linspace(0,80,80)
    ctime=time.time()
    #print(ctime)
    #with MPIExecutor() as executor:
    with MPIExecutor() as pool:
    #with MPICommExecutor(MPI.COMM_WORLD,root=0) as executor:
    #with confu.ProcessPoolExecutor(2) as executor: 
        pool.workers_exit()
        jobs=list(pool.map(gen,from_iterable(y)))
        for job in jobs:
            print(f"value={job[0]} at rank {job[1]} running on {job[2]}\n")
    etime=time.time()
    #print(f'time difference:{etime-ctime}')
if __name__ == '__main__':
    main()
