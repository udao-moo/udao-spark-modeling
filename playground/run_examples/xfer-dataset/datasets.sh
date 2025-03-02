# to job
bash mlp_xfer_dataset.sh job "cache_and_ckp/job_100000x1/q_compile/ea0378f56dcf" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 None

# node17
lr=3e-3
bash mlp_xfer_dataset_ft.sh job "cache_and_ckp/job_100000x1/q_compile/ea0378f56dcf" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 $lr None

# node18
lr=3e-3
bash mlp_xfer_dataset_ft.sh job "cache_and_ckp/job_100000x1/q_compile/ea0378f56dcf" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 1000 28 "1 0" 0.1 false 1e-2 $lr None

# to tpc-h (ID)
for layer in 1 2 3; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 None ft $layer
done

# to tpc-h (OOD)
for fold in 1 2; do # node16
for layer in 1 2 3; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer
done
done

for fold in 3 4; do # node17
for layer in 1 2 3; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer
done
done

for fold in 5 6; do # node8
for layer in 1 2 3; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer
done
done

for fold in 7 8; do # node9
for layer in 1 2 3; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer
done
done

for fold in 9 10; do # node10
for layer in 1 2 3; do
bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold ft $layer
done
done

## finetune

# to tpc-h (ID) # node 8
for lr in 3e-5 1e-4 3e-4; do
bash mlp_xfer_dataset_ft.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 $lr None
done
