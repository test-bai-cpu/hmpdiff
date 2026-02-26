# TODO
1. Now the ATC dataset, the epoch time is not fully consistent, can use the interpolation, to make the every time step is fully consistent, like each step is 
exactly 1s.

# Install the env

conda create -n hmpdiff python=3.10
conda activate hmpdiff


# Run the env
cd research/diffusion/HMP_diffusion/
conda activate hmpdiff

# Run the experiments
python3 train_k.py > logs/log_feb_23.txt  2>&1

# screen
go into the screen: screen -R hmpdiff
exit the screen: ctrl + A + D

# Results
Epoch 100:
Test ADE (original units): 5.0075
Test FDE (original units): 10.1112





# Experiments version

v8-20-k-mod, Feb 23, sigma=0.2, lambda_gt * L_gt + lambda_smooth * L_smooth + lambda_mod * L_mod. lambda_gt = 1.0, lambda_mod = 1e-3, lambda_smooth = 1e-3






