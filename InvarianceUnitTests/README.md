# Comparisons on linear unit-tests 

## Run
For experiments with three environments
```bash
python3 scripts/sweep.py \
    --models ERM IRMv1 IB_ERM IB_IRM \
    --num_iterations 10000 \
    --datasets Example1 Example1s Example2 Example2s Example3 Example3s \
    --dim_inv 5 --dim_spu 5 \
    --n_envs 3 \
    --num_data_seeds 50 --num_model_seeds 20 \
    --output_dir test_results/3envs
```

For experiments with six environments
```bash
python3 scripts/sweep.py \
    --models ERM IRMv1 IB_ERM IB_IRM \
    --num_iterations 10000 \
    --datasets Example1 Example1s Example2 Example2s Example3 Example3s \
    --dim_inv 5 --dim_spu 5 \
    --n_envs 6 \
    --num_data_seeds 50 --num_model_seeds 20 \
    --output_dir test_results/6envs
```

## Analyze
`test_peak` means the number of test sample used for validation. 
It can take values in `0, 20, 100, 500`.
`0` means using data from train distribution for validation.

```bash
python scripts/collect_results.py test_results/3envs --test_peak 0
python scripts/collect_results.py test_results/6envs --test_peak 0
```

Our code is based on the [InvarianceUnitTests](https://github.com/facebookresearch/InvarianceUnitTests) suite.