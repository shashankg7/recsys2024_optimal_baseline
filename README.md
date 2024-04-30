# RecSys 2024 full paper submission id 92 Optimal Baseline Corrections for Off-Policy Contextual Bandits	

==============================

Anonymous source code for running the experiments for the RecSys 2024 full paper submission number 92 titled, "Optimal Baseline Corrections for Off-Policy Contextual Bandits".

## Disclaimer
The codebase for reproducing the results from the submission builds on top of the Open Bandit Pipeline (https://github.com/st-tech/zr-obp). 

## Reproducing the plots

We have included a Jupyter notebook (Paper Plots.ipynb) along with the temp. results files from different runs of off-policy learning and evaluation experiments to reproduce the figures (Figure 1,2,3,4) from the paper.

## Additional results -- empirical variance for off-policy evaluation

We have included additional results from the Section 5.4 with empirical variance of different estimators for off-policy evaluation task (analogous to Figure 4 from the paper). 

The results are present in the file "appendix/Evaluation_variance.pdf" in the current folder.


Steps to run the code: 
-----------------------

1) Create a new conda envioronment and install the dependencies for the project via the requirements.txt file

```
conda create --name recsys_2024
pip3 install -r requirements.txt
```

## Off-policy Evaluation

1) In the script:
```
./examples/obd/evaluate_off_policy_synt_slurm.py
```
modify "base_path" and "log_path" variables with the absolute path to the parent folder of the code. 

2) For off-policy evaluation with an inverse temperature (beta) parameter of 1 for the softmax behavior policy with logged data size 1000000, run the following command:
```
python examples/obd/evaluate_off_policy_estimators_synt_slurm.py --iteration 1 --N 1000000 --beta 1
```

## Off-policy learning -- mini-batch

3) For OPL with mini-batch setup, we need to tune BanditNet first. To tune BanditNet with a batch size of 1024, learning rate of 0.01 and with 250K logged data size, run the following script:

```
python examples/opl/tune_banditnet.py --n_rounds 250000 --optimizer adam --batch_size 1024 --learning_rate 0.01
```

4) Launch the following script to run all models

```
python examples/opl/evaluate_off_policy_learners.py --n_rounds 250000 --optimizer sgd --batch_size 1024 --learning_rate 0.1 --random_state 12345
```

## Off-policy learning -- full-batch

5) Run the following command:

```
python examples/opl/evaluate_off_policy_learners_fullbatch.py --n_rounds 250000 --optimizer adam --batch_size 128 --learning_rate 0.1 --random_state 12345 --epoch 500
```

## Off-policy learning/evaluation on slurm

For OPE, update the paths in the ope.job file and launch the script, and for OPL experiments, first run 1) tune_banditnet.job, followed by: 2) opl.job (mini-batch), opl_fullbatch.job (full-batch).











