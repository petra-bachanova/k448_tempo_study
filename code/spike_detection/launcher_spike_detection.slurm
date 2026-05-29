#!/bin/bash
# --------------------------------------------------------------------------------------------------------
#SBATCH -J spike_detection                                                  # Job name
#SBATCH -o /dartfs-hpc/rc/lab/E/ECoG/k448_tempo_study/temp_out_%j.txt      # Name of stdout output file
#SBATCH -e /dartfs-hpc/rc/lab/E/ECoG/k448_tempo_study/temp_err_%j.txt      # Name of stderr output file
#SBATCH -N 1                                                                # Total # of nodes
#SBATCH -n 1                                                                # Total # of mpi tasks
#SBATCH -c 8                                                                # Allocate # of CPUs per task
#SBATCH --mem=32G                                                           # Allocate memory
#SBATCH -t 24:00:00                                                         # Run time (hh:mm:ss)
#SBATCH --mail-type=END,FAIL                                                # Get email on job end or fail
#SBATCH --mail-user=petra.bachanova.gr@dartmouth.edu                        # Where to send email
# --------------------------------------------------------------------------------------------------------

export SUBJECT="sub-32"
export SESSION="ses-01"
export FULLID="${SUBJECT}_${SESSION}"

PROJECT_PATH="/dartfs-hpc/rc/lab/E/ECoG/k448_tempo_study"
LAUNCHER_OUTPUT_DIR="${PROJECT_PATH}/derivatives/spike_detection/launcher_output"

mv ${PROJECT_PATH}/temp_out_${SLURM_JOB_ID}.txt ${LAUNCHER_OUTPUT_DIR}/${FULLID}_spike_detection_${SLURM_JOB_ID}_out.txt
mv ${PROJECT_PATH}/temp_err_${SLURM_JOB_ID}.txt ${LAUNCHER_OUTPUT_DIR}/${FULLID}_spike_detection_${SLURM_JOB_ID}_err.txt

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate mne_bids

echo "FULLID: $FULLID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

python3 ${PROJECT_PATH}/code/spike_detection/spike_detection_run.py
