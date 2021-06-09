# Comparisons on AC-CMNIST

## Run
Two examples:

```bash
cd ./experiments/cmnist/irm_baseline/
python val_main.py --n_restarts 30 --val 0.05 --test_valid 0
python val_main.py --n_restarts 30 --val 0.20 --test_valid 1
```
where `--val` controls the ratio of data used for validation, 
`--test_valid` decides whether to split out part of test data for validation.

The checkpoints are not stored, 
and the final results will be printed after the whole run. 

Our code is based on the [this repo](https://github.com/facebookresearch/InvariantRiskMinimization). 