fold=None
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=1
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=2
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=3
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=4
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=5
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=6
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=7
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=8
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=9
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=10
njoins=3
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000 $njoins

fold=1
bash mlp_abalation.sh tpcds+job q_compile run_graph_gtn_sk_mlp 4 256 1 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold 30 job-ext 27000
