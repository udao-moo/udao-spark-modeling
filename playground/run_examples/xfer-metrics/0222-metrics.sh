# TPCDS (from GTN)
fold=None
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins" "graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4

for fold in 1 2 3; do
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf-${fold}_30_ext_27000_ext2_30000_upto4joins" "graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4
done

for fold in 4 5 6; do
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf-${fold}_30_ext_27000_ext2_30000_upto4joins" "graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4
done

for fold in 7 8 9 10; do
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf-${fold}_30_ext_27000_ext2_30000_upto4joins" "graph_gtn_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4
done

# TPCDS (from GAT-NPE)
fold=None
njoins=4
bash mlp_abalation.sh tpcds+job q_compile run_graph_gat_no_pe_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 $njoins
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf_30_ext_27000_ext2_30000_upto4joins" "graph_gat_no_pe_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4

for fold in 1 2; do
njoins=4
bash mlp_abalation.sh tpcds+job q_compile run_graph_gat_no_pe_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 $njoins
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf-${fold}_30_ext_27000_ext2_30000_upto4joins" "graph_gat_no_pe_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4
done

for fold in 3 4; do
  njoins=4
bash mlp_abalation.sh tpcds+job q_compile run_graph_gat_no_pe_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 $njoins
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf-${fold}_30_ext_27000_ext2_30000_upto4joins" "graph_gat_no_pe_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4
done

for fold in 5 6; do
  njoins=4
bash mlp_abalation.sh tpcds+job q_compile run_graph_gat_no_pe_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 $njoins
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf-${fold}_30_ext_27000_ext2_30000_upto4joins" "graph_gat_no_pe_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4
done

for fold in 7 8; do
  njoins=4
 bash mlp_abalation.sh tpcds+job q_compile run_graph_gat_no_pe_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 $njoins
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf-${fold}_30_ext_27000_ext2_30000_upto4joins" "graph_gat_no_pe_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4
done

for fold in 9 10; do
  njoins=4
bash mlp_abalation.sh tpcds+job q_compile run_graph_gat_no_pe_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 $njoins
bash mlp_xfer_metric.sh tpcds+job "cache_and_ckp/tpcds+job_102x490+100000x1+ext_27371x1+ext2_star_joins_5518x10/q_compile/ea0378f56dcf-${fold}_30_ext_27000_ext2_30000_upto4joins" "graph_gat_no_pe_sk_mlp1342c11f7ff5/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 $fold 30 job-ext 27000 None 42 None tpcds-ext-star-joins 30000 4
done
