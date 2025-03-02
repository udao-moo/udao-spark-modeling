# ------------------------------------------
# TPC-DS
# ------------------------------------------

# QPPNet (node1)
bash qppnet_train.sh tpcds q_compile 300 28 3e-3

# TCNN (node2): respect original: readout: max, tcnn_hidden: 256, embedder_output_size: 64)
bash tcnn_train.sh tpcds q_compile run_tree_cnn_sk_mlp 4 256 300 28 3e-3 "1 0" None 256 64

# TLSTM (node5)
bash lstm_train.sh tpcds q_compile run_tree_lstm_sk_mlp 4 256 300 28 3e-3 "1 0"

# RAAL (node7)
lr=3e-3
bash mlp_abalation.sh tpcds q_compile run_graph_raal_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None

# QueryFormer (FIXME)
lr=3e-3
bash mlp_abalation.sh tpcds q_compile run_graph_qf_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None

# GTN (done)
lr=3e-3
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None


for fold in 1 2 3 4 5; do
for lr in 3e-3 1e-3 1e-2; do
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold
done
done

for fold in 6 7 8 9 10; do
for lr in 3e-3 1e-3 1e-2; do
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold
done
done

# ------------------------------------------
# TPC-H
# ------------------------------------------

# QPPNet (Done)
bash qppnet_train.sh tpch q_compile 300 28 3e-3

# TCNN (Done): respect original: readout: max, tcnn_hidden: 256, embedder_output_size: 64)
bash tcnn_train.sh tpch q_compile run_tree_cnn_sk_mlp 4 256 300 28 3e-3 "1 0" None 256 64

# TLSTM (Done)
bash lstm_train.sh tpch q_compile run_tree_lstm_sk_mlp 4 256 300 28 3e-3 "1 0"

# RAAL (node14)
lr=3e-3
bash mlp_abalation.sh tpch q_compile run_graph_raal_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None

# QueryFormer (FIXME)
lr=3e-3
bash mlp_abalation.sh tpch q_compile run_graph_qf_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None

# GTN (Done)
lr=3e-3
bash mlp_abalation.sh tpch q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None


# ------------------------------------------
# JOB
# ------------------------------------------
# QPPNet (node17)
bash qppnet_train.sh job q_compile 300 28 3e-3

# TCNN (node19): respect original: readout: max, tcnn_hidden: 256, embedder_output_size: 64)
bash tcnn_train.sh job q_compile run_tree_cnn_sk_mlp 4 256 300 28 3e-3 "1 0" None 256 64

# TLSTM (Done)
bash lstm_train.sh job q_compile run_tree_lstm_sk_mlp 4 256 300 28 3e-3 "1 0"

# RAAL (node21-1)
lr=3e-3
bash mlp_abalation.sh job q_compile run_graph_raal_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None

# QueryFormer (FIXME)
lr=3e-3
bash mlp_abalation.sh job q_compile run_graph_qf_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None

# GTN (Done)
lr=3e-3
bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 None


bash job_unit.sh 3e-3 qf  # node1
bash tpc_unit.sh None 3e-3 tpcds qf # node2
bash tpc_unit.sh 1 3e-3 tpcds qf # node3
bash tpc_unit.sh 2 3e-3 tpcds qf # node4
bash tpc_unit.sh 3 3e-3 tpcds qf # node5

bash tpc_unit.sh 8,9 3e-3 tpcds qppnet # node6
bash tpc_unit.sh 8,9 3e-3 tpcds tcnn # node7
bash tpc_unit.sh 10 3e-3 tpcds qppnet,tcnn # node8
bash tpc_unit.sh 8,9 3e-3 tpcds tlstm # node9
bash tpc_unit.sh 10 3e-3 tpcds tlstm,qf # node10
bash tpc_unit.sh 4,5 3e-3 tpcds qf # node11
bash tpc_unit.sh 6,7 3e-3 tpcds qf # node12
bash tpc_unit.sh 8,9 3e-3 tpcds qf # node14

bash tpc_unit.sh 5,10 3e-3 tpch tcnn # node15
bash tpc_unit.sh None,1,2 3e-3 tpch qf # node16
bash tpc_unit.sh 3,4 3e-3 tpch qf # node17
bash tpc_unit.sh 5,6 3e-3 tpch qf # node18
bash tpc_unit.sh 7,8 3e-3 tpch qf # node19
bash tpc_unit.sh 9,10 3e-3 tpch qf # node20

 # node13

# GCN
lr=3e-3
bash gcn_train.sh tpch q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 $lr 3 None

# GCN
lr=3e-3
bash gcn_train.sh tpcds q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 $lr 3 None

# GCN
lr=3e-3
bash gcn_train.sh job q_compile run_graph_conv_net_sk_mlp 4 256 300 28 "1 0" 0.1 1e-2 128 $lr 3 None
