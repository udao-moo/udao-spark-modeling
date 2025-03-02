# ------ ID
# ALL
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node8
# 1-HOT
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node9
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node11
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node12
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "hist" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node2
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node15
# STATS-OFF
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node3
# 1-OFF
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc hist " 0.1 false 1e-2 128 $lr 3 4 0.1; done # node4
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo op_enc bitmap " 0.1 false 1e-2 128 $lr 3 4 0.1; done # node6
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node18
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "type cbo hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node20
for lr in 3e-3 1e-3 1e-2; do bash mlp_abalation.sh job q_compile run_graph_gtn_sk_mlp 4 256 300 28 "1 0" "cbo op_enc hist bitmap" 0.1 false 1e-2 128 $lr 3 4 0.1; done # node14,16,17

## OOD
