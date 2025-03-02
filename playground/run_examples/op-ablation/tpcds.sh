# ------ ID
# ALL
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done
# 1-HOT
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node14
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node15
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node16
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "hist" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node17
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node18
# STATS-OFF
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node19
# 1-OFF
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist " 0.1 false 1e-2 128 $lr 3 4 0.1; done # node20
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc bitmap " 0.1 false 1e-2 128 $lr 3 4 0.1; done # node21
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node21
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node19
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node7

# ------ OOD

# ALL
for lr in 3e-3 1e-3 1e-2; do for fold in 1 2 3 4 5 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node17
# 1-HOT
for lr in 3e-3 1e-3 1e-2; do for fold in 1 2 3 4 5 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node21
for lr in 3e-3 1e-3 1e-2; do for fold in 1 2 3 4 5 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node21
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "hist" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "hist" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
# STATS-OFF
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
# 1-OFF
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist " 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist " 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc bitmap " 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc bitmap " 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done;  done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done;  done
for lr in 3e-3; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 3e-3; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
------

for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node5
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node6
for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "hist" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node8
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "hist" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node10
for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node11
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node12
# STATS-OFF
for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node14
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node19
# 1-OFF
for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist " 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node20
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist " 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node13
for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc bitmap " 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node2
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc bitmap " 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node4
for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node9
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done # node18
for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done;  done # node16
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done;  done
for lr in 1e-3 1e-2; do for fold in 1 2 3 4 5; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
for lr in 1e-3 1e-2; do for fold in 6 7 8 9 10; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done

for fold in 1 2 3 4 5; do
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold
done
for fold in 6 7 8 9 10; do
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold
done

fold=4
seed=0
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold None None None $seed
fold=4
seed=1
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold None None None $seed
fold=4
seed=2
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold None None None $seed
fold=9
seed=0
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold None None None $seed
fold=9
seed=1
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold None None None $seed
fold=9
seed=2
bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 3e-3 3 4 0.1 $fold None None None $seed



fold=1
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=2
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=3
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=4
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=5
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=6
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=7
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=8
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=9
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=10
for lr in 1e-3; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=1
for lr in 1e-2; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=2
for lr in 1e-2; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=3
for lr in 1e-2; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=4
for lr in 1e-2; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=5
for lr in 1e-2; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=6
for lr in 1e-2; do for op in "type" "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=7 # todo
for lr in 1e-2; do for op in "type"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=8 # todo
for lr in 1e-2; do for op in "type"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=9 # todo
for lr in 1e-2; do for op in "type"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=10 # todo
for lr in 1e-2; do for op in "type"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=7 # todo
for lr in 1e-2; do for op in "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=8 # todo
for lr in 1e-2; do for op in "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=9 # todo
for lr in 1e-2; do for op in "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
fold=10 # todo
for lr in 1e-2; do for op in "cbo"; do bash mlp_abalation.sh tpcds q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" ${op} 0.1 false 1e-2 128 $lr 3 4 0.1 $fold; done; done
