# index creations

for fold in 1 2; do
  for peek in 1 5 10 20 30 40 50 60 70 80; do
    lr=3e-3
    bash mlp_abalation_perc.sh tpch q_compile run_graph_gtn_sk_mlp 4 256 1 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold $peek
    bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}_peek_${peek}_percents" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None false $peek
  done
done


for fold in 3 4; do
  for peek in 1 5 10 20 30 40 50 60 70 80; do
    lr=3e-3
    bash mlp_abalation_perc.sh tpch q_compile run_graph_gtn_sk_mlp 4 256 1 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold $peek
    bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}_peek_${peek}_percents" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None false $peek
  done
done


for fold in 5 6; do
  for peek in 1 5 10 20 30 40 50 60 70 80; do
    lr=3e-3
    bash mlp_abalation_perc.sh tpch q_compile run_graph_gtn_sk_mlp 4 256 1 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold $peek
    bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}_peek_${peek}_percents" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None false $peek
  done
done

for fold in 7 8; do
  for peek in 1 5 10 20 30 40 50 60 70 80; do
    lr=3e-3
    bash mlp_abalation_perc.sh tpch q_compile run_graph_gtn_sk_mlp 4 256 1 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold $peek
    bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}_peek_${peek}_percents" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None false $peek
  done
done


for fold in 9 10; do
  for peek in 1 5 10 20 30 40 50 60 70 80; do
    lr=3e-3
    bash mlp_abalation_perc.sh tpch q_compile run_graph_gtn_sk_mlp 4 256 1 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold $peek
    bash mlp_xfer_dataset.sh tpch "cache_and_ckp/tpch_22x2273/q_compile/ea0378f56dcf-${fold}_peek_${peek}_percents" "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins/graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "1 0" 0.1 false 1e-2 3e-3 $fold None None false $peek
  done
done
