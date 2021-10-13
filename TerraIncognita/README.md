# Reproduce the TerraIncognita results 

This folder should help you reproduce the table 4 for ERM, IRM, IB-ERM and IB-IRM on TerraIncognita

You need to download the dataset with
```sh
python3 -m domainbed.scripts.download \
       --data_dir=/my/datasets/path
```

To launch the sweep
```sh
python3 -m domainbed.scripts.sweep launch\
	--algorithms ERM IRM IB_ERM_F_C IB_IRM_F_C\
	--datasets TerraIncognita\
       	--data_dir=/my/datasets/path\
       	--output_dir=/my/sweep/output/path\
       	--command_launcher local\
	--unique_test_env 0\
	--skip_confirmation
```

To view the results of your sweep:

````sh
python -m domainbed.scripts.collect_results\
       --input_dir=/my/sweep/output/path
````
