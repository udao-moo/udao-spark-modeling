# JOB

# GAT
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-4 None 30 job-ext 27000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 1e-4 None 30 job-ext 27000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 None 30 job-ext 27000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 1e-3 None 30 job-ext 27000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 1e-2 None 30 job-ext 27000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 3e-2 3e-3 None 30 job-ext 27000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-1 3e-3 None 30 job-ext 27000

lr=3e-3
eps=1000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 $eps 28 "0 1" 0.1 false 1e-2 $lr None 30 job-ext 27000

lr=1e-3
eps=1000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 $eps 28 "0 1" 0.1 false 1e-2 $lr None 30 job-ext 27000

lr=3e-4
eps=1000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 $eps 28 "0 1" 0.1 false 1e-2 $lr None 30 job-ext 27000

lr=1e-4
eps=1000
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/ea0378f56dcf_30_ext_27000" "graph_gat_no_pe_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 $eps 28 "0 1" 0.1 false 1e-2 $lr None 30 job-ext 27000


# GTN
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/de5cd9deaae1_30_ext_27000" "graph_gtn_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 3e-3 None 30 job-ext 27000

bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/de5cd9deaae1_30_ext_27000" "graph_gtn_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 1000 28 "0 1" 0.1 false 1e-2 3e-3 None 30 job-ext 27000

for lr in 3e-4 1e-4 1e-3 1e-2; do
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/de5cd9deaae1_30_ext_27000" "graph_gtn_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false 1e-2 $lr None 30 job-ext 27000
done

for wd in 3e-3 1e-3 3e-2 1e-1; do
bash mlp_xfer_metric.sh job "cache_and_ckp/job_100000x1+ext_27371x1/q_compile/de5cd9deaae1_30_ext_27000" "graph_gtn_sk_mlpeb9dd956e59b/learning_33645c234894_1_0" 4 256 300 28 "0 1" 0.1 false $wd 3e-3 None 30 job-ext 27000
done



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
