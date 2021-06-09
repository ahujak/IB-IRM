# Comparisons on CS-CMNIST

## Run
Two examples:

```bash
python -m domainbed.scripts.sweep_train --holdout_fraction 0.2
python -m domainbed.scripts.sweep_train --holdout_fraction 0.2 --test_val
```
where `--holdout_fraction` controls the ratio of data used for validation, 
`--test_val` decides whether to set validation distribution to be same as test distribution.

The checkpoints are not stored, 
and the final results will be printed after the whole run. 

Our code is based on the [DomainBed](https://github.com/facebookresearch/DomainBed) suite.    