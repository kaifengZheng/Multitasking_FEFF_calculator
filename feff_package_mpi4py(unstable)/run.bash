#!/usr/bin/env bash
#rm -rf /gpfs/scratch/kaifzheng/FEFF/*
rm -r out*
rm array*
rm -r FEFF_inp
sbatch run.slurm
