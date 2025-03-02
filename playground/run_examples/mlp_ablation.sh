# Assigning positional parameters to variables
bm=$1
q_type=$2
file_choice=$3
nlayers=$4
hdim=$5
eps=$6
nworkers=$7
loss_weights=${8:-None}
ops=${9:-"type cbo op_enc hist bitmap"}
dropout=${10:-0.1}
bn=${11:-true}
wd=${12:-1e-2}
osize=${13:-32}
lr=${14:-1e-3}
gtn_n_layers=${15:-2}
gtn_n_heads=${16:-2}
gtn_dp=${17:-0.0}
fold=${18:-None}
perc=${19:-None}
bm_ext=${20:-None}
ext=${21:-None}

# Default values for other variables
lsize=8
vsize=16
bs=1024
act=relu

command="python -m ${file_choice} \
--benchmark $bm \
--q_type $q_type \
--init_lr $lr \
--min_lr 1e-7 \
--weight_decay $wd \
--epochs $eps \
--batch_size $bs \
--num_workers $nworkers \
--lpe_size $lsize \
--output_size $osize \
--vec_size $vsize \
--n_layers $nlayers \
--hidden_dim $hdim \
--activate $act \
--gtn_n_layers $gtn_n_layers \
--gtn_n_heads $gtn_n_heads \
--gtn_dropout $gtn_dp \
--op_groups $ops \
--dropout $dropout \
--pos_encoding_dim $lsize"

if [[ "$bn" = "true" ]]; then
    command="$command --use_batchnorm"
fi

if [[ "$fold" != "None" ]]; then
    command="$command --fold $fold"
fi

if [[ "$perc" != "None" ]]; then
    command="$command --data_percentage $perc"
fi

if [[ "$bm_ext" != "None" ]]; then
    command="$command --benchmark_ext $bm_ext"
fi

if [[ "$ext" != "None" ]]; then
    command="$command --ext_data_amount $ext"
fi

if [[ "$loss_weights" = "None" ]]; then
    echo $command
    $command
else
    command="$command --loss_weights $loss_weights"
    echo $command
    $command
fi
