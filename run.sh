#!/bin/bash
. path.sh

#input choice 
stage=$1        # <=1: preparation <=2: training <=3: generating <=4: evaluating 
fea_type=$2     # "vggish" OR "i3d_flow" OR "vggish i3d_flow"
fea_names=$3    # vggish OR i3dflow OR vggish+i3dflow 
num_epochs=$3   # e.g. 20 
warmup_steps=$5 # e.g. 9660
dropout=$6      # e.g. 0.2

# data setting 
batch_size=32                   # number of dialogue instances in each batch 
max_length=256                  # batch size is reduced if len(input_feature) >= max_length
include_caption=caption,summary # concatenate caption and summary together 
sep_caption=1                   # separate caption from history 
max_his_len=-1                  # -1 1 2 ... 10; -1 for all dialogue turns possible 
merge_source=0                  # concatenate history(+caption) and query together as one single source sequence
decode_data=off                 # use official data for testing 
undisclosed_only=1              # only decode undisclosed dialogue turns in official data 
data_root=data                  # directory of data
fea_dir=$data_root
fea_file="<FeaType>/<ImageID>.npy" 

# model setting 
sep_his_embed=0         # separate history embedding from source sequence embedding 
sep_cap_embed=0         # separate caption embedding from source sequence embedding 
nb_blocks=6             # number of attention blocks 
d_model=512             # feature dimensions 
d_ff=$(( d_model*4 ))   # feed-forward hidden layer 
att_h=8                 # attention heads 
# auto-encoder setting  
diff_encoder=1          # use different query encoder weights in auto-encoder   
diff_embed=0            # use different query embedding weights in auto-encoder
diff_gen=0              # use different generator in auto-encoder 
auto_encoder_ft=query   # features to be auto-encoded e.g. query, caption, summary  

# training setting
decode_style=beam_search    # beam search OR greedy 
cut_a=1                     # 0: none OR 1: randomly truncated responses for token-level decoding simulation in training 
loss_l=1                    # lambda in loss function 
seed=1                      # random seed 
model_prefix=mtn                                                # model name 
expid=${fea_names}_warmup${warmup_steps}_epochs${num_epochs}_dropout${dropout}    # output folder name 
expdir=exps/${expid}                                            # output folder directory 

# generation setting 
beam=5                  # beam width
penalty=1.0             # penalty added to the score of each hypothesis
nbest=5                 # number of hypotheses to be output
model_epoch=best        # model epoch number to be used
report_interval=100     # step interval to report losses during training 

echo Stage $stage Exp ID $expid

workdir=`pwd`
labeled_test=''
train_set=$data_root/train_set4DSTC7-AVSD.json
valid_set=$data_root/valid_set4DSTC7-AVSD.json
test_set=$data_root/test_set.json
labeled_test=$data_root/test_set.json
eval_set=${labeled_test}
if [ $decode_data = 'off' ]; then
  test_set=$data_root/test_set4DSTC7-AVSD.json
  labeled_test=$data_root/lbl_test_set4DSTC7-AVSD.json
  eval_set=${labeled_test}
  if [ $undisclosed_only -eq 1 ]; then
    eval_set=$data_root/lbl_undisclosedonly_test_set4DSTC7-AVSD.json 
  fi
fi
echo Exp Directory $expdir 

. utils/parse_options.sh || exit 1;

# directory and feature file setting
enc_psize_=`echo $enc_psize|sed "s/ /-/g"`
enc_hsize_=`echo $enc_hsize|sed "s/ /-/g"`
fea_type_=`echo $fea_type|sed "s/ /-/g"`

# command settings
train_cmd=""
test_cmd=""
gpu_id=`utils/get_available_gpu_id.sh`

set -e
set -u
set -o pipefail

# preparation
if [ $stage -le 1 ]; then
    echo -------------------------
    echo stage 1: preparation 
    echo -------------------------
    echo setup ms-coco evaluation tool
    if [ ! -d utils/coco-caption ]; then
        git clone https://github.com/tylin/coco-caption utils/coco-caption
        patch -p0 -u < utils/coco-caption.patch
    else
        echo Already exists.
    fi
fi

# training phase
mkdir -p $expdir
if [ $stage -le 2 ]; then
    echo -------------------------
    echo stage 2: model training
    echo -------------------------
    python train.py \
      --gpu $gpu_id \
      --fea-type $fea_type \
      --train-path "$fea_dir/$fea_file" \
      --train-set $train_set \
      --valid-path "$fea_dir/$fea_file" \
      --valid-set $valid_set \
      --num-epochs $num_epochs \
      --batch-size $batch_size \
      --max-length $max_length \
      --model $expdir/$model_prefix \
      --rand-seed $seed \
      --report-interval $report_interval \
      --nb-blocks $nb_blocks \
      --include-caption $include_caption \
      --max-history-length $max_his_len \
      --separate-his-embed $sep_his_embed \
      --separate-caption $sep_caption \
      --merge-source $merge_source \
      --separate-cap-embed $sep_cap_embed \
      --warmup-steps $warmup_steps \
      --nb-blocks $nb_blocks \
      --d-model $d_model \
      --d-ff $d_ff \
      --att-h $att_h \
      --dropout $dropout \
      --cut-a $cut_a \
      --loss-l ${loss_l} \
      --diff-encoder ${diff_encoder} \
      --diff-embed ${diff_embed} \
      --auto-encoder-ft ${auto_encoder_ft} \
      --diff-gen ${diff_gen}  
fi

# testing phase
if [ $stage -le 3 ]; then
    echo -----------------------------
    echo stage 3: generate responses
    echo -----------------------------
    if [ $decode_data = 'off' ]; then
        fea_file="<FeaType>_testset/<ImageID>.npy"
    fi
    for data_set in $test_set; do
        echo start response generation for $data_set
        target=$(basename ${data_set%.*})
        result=${expdir}/result_${target}_b${beam}_p${penalty}_${decode_style}_undisclosed${undisclosed_only}.json
        test_log=${result%.*}.log
        python generate.py \
          --gpu $gpu_id \
          --test-path "$fea_dir/$fea_file" \
          --test-set $data_set \
          --model-conf $expdir/${model_prefix}.conf \
          --model $expdir/${model_prefix}_${model_epoch} \
          --beam $beam \
          --penalty $penalty \
          --nbest $nbest \
          --output $result \
          --decode-style ${decode_style} \
          --undisclosed-only ${undisclosed_only} \
          --labeled-test ${labeled_test}
         #|& tee $test_log
    done
fi

# scoring only for validation set
if [ $stage -le 4 ]; then
    echo --------------------------
    echo stage 4: score results
    echo --------------------------
    for data_set in $eval_set; do
        echo start evaluation for $data_set
        save_target=$(basename ${test_set%.*})
        target=$(basename ${data_set%.*})
        result=${expdir}/result_${save_target}_b${beam}_p${penalty}_${decode_style}_undisclosed${undisclosed_only}.json
        reference=${result%.*}_ref.json
        hypothesis=${result%.*}_hyp.json
        result_eval=${result%.*}.eval
        echo Evaluating: $result
        utils/get_annotation.py -s data/stopwords.txt $data_set $reference
        utils/get_hypotheses.py -s data/stopwords.txt $result $hypothesis
        python2 utils/evaluate.py $reference $hypothesis >& $result_eval
        echo Wrote details in $result_eval
        echo "--- summary ---"
        awk '/^(Bleu_[1-4]|METEOR|ROUGE_L|CIDEr):/{print $0; if($1=="CIDEr:"){exit}}'\
            $result_eval
        echo "---------------"
    done
fi
