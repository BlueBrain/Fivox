#!/bin/bash
#SBATCH --job-name="voxelize"
#SBATCH --time=06:00:00
#SBATCH --partition=prod
#SBATCH --account=proj3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# Just a template, modify as needed. Example launch (with tcsh):
# foreach i ( `seq 0 63` )
# sbatch $PWD/sbatch.sh $i 64

/gpfs/bbp.cscs.ch/scratch/gss/viz/eilemann/config.bbp/release/bin/voxelize --volume 'fivoxspikes:///gpfs/bbp.cscs.ch/project/proj3/resources/circuits/3M_neuron/BlueConfig?functor=field,duration=1.25,showProgress=1' -t 10 -s 6144 -d char -o /gpfs/bbp.cscs.ch/project/proj3/resources/volumes/3M_neuron_6K --decompose "$1 $2"
