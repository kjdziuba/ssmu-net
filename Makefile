.PHONY: preprocess train train-debug eval audit dfir report visualize all clean

# Data preprocessing
preprocess: ; python scripts/run_preprocess.py

# Training
train: preprocess ; python scripts/run_train.py
train-debug: preprocess ; python scripts/run_train.py --debug

# Evaluation (requires checkpoint path)
eval: ; python scripts/run_eval.py --checkpoint outputs/models/fold_0/checkpoint_best.pth

# Visualization of preprocessed data
visualize: preprocess ; python scripts/visualize_npz.py --n 10

# Analysis tasks
audit: ; python scripts/run_audit.py
dfir: ; python scripts/run_export_dfir.py
report: eval audit dfir ; python scripts/run_report.py

# Run everything
all: preprocess train eval audit dfir report

# Clean outputs
clean: ; rm -rf outputs/models/* outputs/figures/* outputs/tables/* outputs/logs/*