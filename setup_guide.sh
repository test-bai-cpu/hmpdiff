#!/bin/bash
# ============================================================
# HMP Docker Setup Guide for DeepGear
# ============================================================

# ==========================================
# STEP 1: Upload files to DeepGear
# ==========================================
# Upload your HMP project folder, Dockerfile, and docker-compose.yml
# to the DeepGear machine. Structure should look like:
#
# ~/hmp-docker/
# ├── Dockerfile
# ├── docker-compose.yml
# ├── data/
# │   └── atc/
# │       ├── atc1_normalization_stats.npz
# │       ├── full/
# │       │   ├── atc1_train_split.parquet
# │       │   ├── atc1_val_split.parquet
# │       │   └── atc1_test_split.parquet
# │       └── full_frame_20/
# │           └── ...
# ├── checkpoints/
# ├── logs/
# ├── runs/
# └── HMP/              (your project code)
#     ├── train.py
#     ├── train_k.py
#     ├── eval.py
#     ├── model.py
#     ├── traj_dataset.py
#     └── ...


# ==========================================
# STEP 2: Build the Docker image
# ==========================================
cd ~/hmp-docker
docker compose build


# ==========================================
# STEP 3: Start the container
# ==========================================
docker compose up -d


# ==========================================
# STEP 4: Connect to the container
# ==========================================

# Option A: Direct docker exec (if you're on the DeepGear machine)
docker exec -it hmp-training bash

# Option B: SSH from your local machine
ssh -p 2222 root@<DEEPGEAR_IP>
# Password: hmpproject (change this in the Dockerfile!)


# ==========================================
# STEP 5: Run training inside the container
# ==========================================
cd /workspace/HMP

# Single training run
python train_k.py 5 20 0.2 1 1 0 1

# Or use the experiment script
bash run-exp.sh

# Monitor with TensorBoard (from inside container)
tensorboard --logdir=runs --host=0.0.0.0 --port=6006 &

# Then access TensorBoard from your browser:
# http://<DEEPGEAR_IP>:6006


# ==========================================
# STEP 6: Run evaluation
# ==========================================
python eval.py


# ==========================================
# STEP 7: Access from your local machine
# ==========================================

# SSH tunnel for TensorBoard (run on your local machine):
ssh -L 6006:localhost:6006 -p 2222 root@<DEEPGEAR_IP>
# Then open http://localhost:6006 in your browser

# Copy results back to your local machine:
scp -P 2222 -r root@<DEEPGEAR_IP>:/workspace/HMP/checkpoints ./results/


# ==========================================
# Useful Docker commands
# ==========================================

# Stop container
docker compose down

# Restart container
docker compose restart

# View logs
docker logs hmp-training

# Check GPU inside container
docker exec hmp-training nvidia-smi





docker run --gpus all -it --rm \
  -v $(pwd):/workspace/HMP \
  --shm-size 8g \
  --name hmp-diff \
  hmp-training