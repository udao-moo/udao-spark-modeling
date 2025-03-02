# TPC-H
fold=1
lr=3e-3
bash mlp_abalation.sh tpch q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold
bash lstm_train.sh tpch q_compile run_tree_lstm_sk_mlp 4 256 300 28 3e-3 "1 0" $fold
bash tcnn_train.sh tpch q_compile run_tree_cnn_sk_mlp 4 256 300 28 3e-3 "1 0" $fold 256 64
bash qppnet_train.sh tpch q_compile 300 28 3e-3 $fold


bash tmp_run.sh 6
# RAAL (node14)
lr=3e-3
bash mlp_abalation.sh tpch q_compile run_graph_raal_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None
# QueryFormer (FIXME)
lr=3e-3
bash mlp_abalation.sh tpch q_compile run_graph_qf_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None


# TPC-DS
fold=1
lr=3e-3
bash lstm_train.sh tpcds q_compile run_tree_lstm_sk_mlp 4 256 300 28 3e-3 "1 0" $fold
bash tcnn_train.sh tpcds q_compile run_tree_cnn_sk_mlp 4 256 300 28 3e-3 "1 0" $fold 256 64
bash qppnet_train.sh tpcds q_compile 300 28 3e-3 $fold

bash tmp_run_ds.sh 8
bash tmp_run_ds.sh 9
bash tmp_run_ds.sh 10


unit

fold=$1
lr=$2
bm=$3
model=$4
if [$model == "qf"]; then
bash mlp_abalation.sh tpch q_compile run_graph_qf_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold
elif [$model == "tlstm"]; then
bash lstm_train.sh tpcds q_compile run_tree_lstm_sk_mlp 4 256 300 28 3e-3 "1 0" $fold
elif [$model == "tcnn"]; then
bash tcnn_train.sh tpcds q_compile run_tree_cnn_sk_mlp 4 256 300 28 3e-3 "1 0" $fold 256 64
elif [$model == "qppnet"]; then
bash qppnet_train.sh tpcds q_compile 300 28 3e-3 $fold
else


11 choices, 3 choices,

fold=1
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=2
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=3
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=4
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=5
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=6
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=7
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=8
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=9
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done

fold=10
for bm in tpch tpcds; do bash gcn_train.sh $bm q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 3e-3 3 $fold; done
