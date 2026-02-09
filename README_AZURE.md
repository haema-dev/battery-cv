# Azure ML Cloud Training Guide

## 1. Prerequisites
- **Azure CLI**: Run `az login` in your terminal.
- **Config**: Download `config.json` from your Azure ML Workspace and place it in this folder.
- **Compute**: Ensure a compute cluster named `gpu-cluster` (or similar) exists in your workspace.

## 2. File Structure
- `src/train_entry.py`: The script that runs in the cloud. It extracts your zips and trains the model.
- `src/requirements.txt`: Python libraries needed.
- `submit_job.py`: Run this locally to send the job to Azure.

## 3. How to Run
1. Open terminal in this folder.
2. Install Azure ML SDK (if not already done):
   ```bash
   pip install azureml-sdk
   ```
3. Run:
   ```bash
   python submit_job.py
   ```
4. Click the URL printed in the terminal to watch the training progress in Azure ML Studio.

## 4. Key Configurations
- **Batch Size**: 32 (in `src/train_entry.py`)
- **Max Epochs**: 50 (in `submit_job.py`)
- **Data Path**: Automatically mounts `battery-data` container.
