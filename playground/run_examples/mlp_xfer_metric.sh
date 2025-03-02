# mlp_xfer.sh

# Assigning positional parameters to variables
bm=$1
data_header=$2
embedding_path=$3
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
perc=${14:-None}
ext=${15:-None}
ext_amount=${16:-None}
ext_njoins=${17:-None}
seed=${18:-42}
perc2=${19:-None}
ext2=${20:-None}
ext_amount2=${21:-None}
ext_njoins2=${22:-None}

# Default values for other variables
lsize=8
vsize=16
bs=1024
act=relu
q_type=q_compile

command="python run_xfer_metric.py \
--data_header $data_header \
--embedding_path $embedding_path \
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

if [[ "$fold" != "None" ]]; then
    command="$command --fold $fold"
fi

if [[ "$perc" != "None" ]]; then
    command="$command --data_percentage $perc"
fi

if [[ "$perc2" != "None" ]]; then
    command="$command --data_percentage2 $perc2"
fi

if [[ "$ext" != "None" ]]; then
    command="$command --benchmark_ext $ext --ext_data_amount $ext_amount"
fi

if [[ "$ext2" != "None" ]]; then
    command="$command --benchmark_ext2 $ext2 --ext_data_amount2 $ext_amount2"
fi

if [[ "$ext_njoins" != "None" ]]; then
    command="$command --ext_up_to_n_joins $ext_njoins"
fi

if [[ "$ext_njoins2" != "None" ]]; then
    command="$command --ext_up_to_n_joins2 $ext_njoins2"
fi

if [[ "$loss_weights" = "None" ]]; then
    echo $command
    $command
else
    command="$command --loss_weights $loss_weights"
    echo $command
    $command
fi
