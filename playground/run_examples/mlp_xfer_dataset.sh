# mlp_xfer.sh

# Assigning positional parameters to variables
bm=$1
data_header=$2
ckp_header=$3
nlayers=$4
hdim=$5
eps=$6
nworkers=$7
loss_weights=${8:-None}
dropout=${9:-0.1}
bn=${10:-true}
wd=${11:-1e-2}
lr=${12:-1e-3}
fold=${13:-None}
finetune=${14:-None}
ft=${15:-None}
aug=${16:-false}
peek=${17:-0}
seed=${18:-42}

# Default values for other variables
lsize=8
vsize=16
bs=1024
act=relu
q_type=q_compile

command="python run_xfer_dataset.py \
--data_header $data_header \
--ckp_header $ckp_header \
--benchmark $bm \
--q_type $q_type \
--init_lr $lr \
--min_lr 1e-7 \
--weight_decay $wd \
--epochs $eps \
--batch_size $bs \
--num_workers $nworkers \
--n_layers $nlayers \
--hidden_dim $hdim \
--activate $act \
--dropout $dropout \
--seed $seed"

if [[ "$bn" = "true" ]]; then
    command="$command --use_batchnorm"
fi

if [[ "$aug" = "true" ]]; then
    command="$command --augmented"
fi

if [[ "$fold" != "None" ]]; then
    command="$command --fold $fold --fold_peek_percentage $peek"
fi

if [[ "$finetune" != "None" ]]; then
    command="$command --finetune"
    if [[ "$ft" != "None" ]]; then
        command="$command --finetune_layers $ft"
    fi
fi



if [[ "$loss_weights" = "None" ]]; then
    echo $command
    $command
else
    command="$command --loss_weights $loss_weights"
    echo $command
    $command
fi
