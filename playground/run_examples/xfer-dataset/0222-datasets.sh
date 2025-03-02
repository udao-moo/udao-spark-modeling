# to tpc-h (ID)
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 None None None true
for layer in None 1 2 3 4; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 None ft $layer true
done

# to tpc-h (OOD)
for fold in 1 2; do # node16
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None true
for layer in None 1 2 3 4; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer true
done
done

for fold in 3 4; do # node17
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None true
for layer in None 1 2 3 4; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer true
done
done

for fold in 5 6; do # node8
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None true
for layer in None 1 2 3 4; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer true
done
done

for fold in 7 8; do # node9
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None true
for layer in None 1 2 3 4; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer true
done
done

for fold in 9 10; do # node10
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None true
for layer in None 1 2 3 4; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer true
done
done
