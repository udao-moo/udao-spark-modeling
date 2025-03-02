Purpose of all files

## Spark Modeling

### Training baseline models
- Baselines Model Training
  - `run_graph_avg.py`
  - `run_graph_gtn.py`
  - `run_graph_qf.py`
  - `run_graph_raal.py`
  - `run_tree_lstm.py`
  - `run_qppnet.py`
- Ensemble Model Training
  - `train_ensemble_models.py`
- Failure Classifier Model Training
  - `train_failure_clf.py`


### Evaluation
- Basic Evaluation Test: `eval_model.py`
- Rule Violation Evaluation: `eval_rule_violation.py`

## Spark MOO - optimizer

- `compile_time_hierarchical_optimizer.py` (compile time optimizer)
- `runtime_optimizer.py` (runtime server)
