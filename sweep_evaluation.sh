#!/bin/bash

FAIRSEQ=/private/home/skottur/repos/MTN/

DATE=20180526
SWEEP_NAME=evaluate.generate
JOBSCRIPTS=scripts
mkdir -p ${JOBSCRIPTS}

queue=learnfair

data_root=data_local/simmc2_mtn
fea_dir=$data_root
fea_file="visual_features_resnet15_all_py2.pk"

# training setting
warmup_steps=1000
num_epochs=100
dropout=0
fea_names=resnet

decode_style=beam_search    # beam search OR greedy
model_prefix=mtn                      # model name
# output folder name
expid=${fea_names}_warmup${warmup_steps}_epochs${num_epochs}_dropout${dropout}
expdir=exps/${expid}             # output folder directory


# generation setting
beam=5                  # beam width
penalty=1.0             # penalty added to the score of each hypothesis
nbest=5                 # number of hypotheses to be output
model_epoch=best        # model epoch number to be used
undisclosed_only=0


test_set=$data_root/simmc2_dials_dstc10_devtest.json
labeled_test=$data_root/simmc2_dials_dstc10_devtest.json
target=$(basename ${test_set%.*})
test_log=${result%.*}.lo


SAVE_ROOT=/checkpoint/skottur/logs/mtn/evaluation/${DATE}/${SWEEP_NAME}
mkdir -p stdout stderr

INCREMENT=50
TOTAL_INSTANCES=1500

for (( start_ind=0; start_ind<=${TOTAL_INSTANCES}; start_ind+=${INCREMENT} )); do
    end_ind=$(( $start_ind + $INCREMENT ))
    result=${expdir}/result_${target}_b${beam}_p${penalty}_${decode_style}_undisclosed${undisclosed_only}_start${start_ind}.json
    SAVE=${SAVE_ROOT}.start_ind${start_ind}
    mkdir -p ${SAVE}
    JNAME=${SWEEP_NAME}.start_ind${start_ind}
    SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
    SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm
    extra=""
    echo "#!/bin/sh" > ${SCRIPT}
    echo "#!/bin/sh" > ${SLURM}
    echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
    echo "#SBATCH --output=stdout/${JNAME}.%j" >> ${SLURM}
    echo "#SBATCH --error=stderr/${JNAME}.%j" >> ${SLURM}
    echo "#SBATCH --partition=$queue" >> ${SLURM}
    echo "#SBATCH --time=720" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}
    echo "#SBATCH --gpus-per-task=1" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=1" >> ${SLURM}
    echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
    echo "srun sh ${SCRIPT}" >> ${SLURM}
    echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
    echo "{ " >> ${SCRIPT}
    echo "echo $SWEEP_NAME $BSZ " >> ${SCRIPT}
    echo "nvidia-smi" >> ${SCRIPT}
    echo "cd $FAIRSEQ" >> ${SCRIPT}

    echo python generate.py \
      --test-path "$fea_dir/$fea_file" \
      --test-set $test_set \
      --model-conf $expdir/${model_prefix}.conf \
      --model $expdir/${model_prefix}_${model_epoch} \
      --beam $beam \
      --penalty $penalty \
      --nbest $nbest \
      --output $result \
      --decode-style ${decode_style} \
      --undisclosed-only ${undisclosed_only} \
      --labeled-test ${labeled_test} \
      --start_ind ${start_ind} \
      --end_ind ${end_ind} >> ${SCRIPT}
    echo "nvidia-smi" >> ${SCRIPT}
    echo "kill -9 \$\$" >> ${SCRIPT}
    echo "} & " >> ${SCRIPT}
    echo "child_pid=\$!" >> ${SCRIPT}
    echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
    echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
    echo "while true; do     sleep 1; done" >> ${SCRIPT}
    sbatch ${SLURM}
done
