# mlp_train.sh

# bash mlp_abalation.sh job q_compile run_graph_gat_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 ${fold} $perc job-ext 27000
# bash mlp_abalation.sh job q_compile run_graph_gat_sk_mlp 4 256 300 28 "0 1" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 ${fold} $perc job-ext 27000
# graph_gat_sk_mlpfd5412a98597

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
peek=${19:-0}
perc=${20:-None}
ext=${21:-None}
ext_amount=${22:-None}
ext_njoins=${23:-None}
seed=${24:-42}
perc2=${25:-None}
ext2=${26:-None}
ext_amount2=${27:-None}
ext_njoins2=${28:-None}

# Default values for other variables
lsize=8
vsize=16
bs=1024
act=relu

command="python ${file_choice}.py \
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
--pos_encoding_dim $lsize \
--seed $seed"

if [[ "$bn" = "true" ]]; then
    command="$command --use_batchnorm"
fi

if [[ "$fold" != "None" ]]; then
    command="$command --fold $fold --fold_peek_percentage $peek"
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
