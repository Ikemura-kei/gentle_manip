#! /bin/bash

#SBATCH -N 1
#SBATCH -t 22:00:00 # Requested walltime
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err


# Do stuff
cd /nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
echo do stuff