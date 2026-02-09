# 🔋 Battery Defect Detection Pipeline

This repository is configured for automated Azure ML training using GitHub Actions.

## 📂 Project Structure

- `.github/workflows/train.yml`: CI/CD Pipeline Definition (Do not edit)
- `command/`: Backup for command scripts (Do not edit)
- `env/`: Backup for environment configs (Do not edit)
- `src/`: Source code directory
    - `index.py`: Main entry point for training (renamed from `train_entry.py`)
    - `conda_env.yml`: Environment specification
- `train-job.yml`: Azure ML Job Specification (Edit if environment changes)

## 🚀 How to Run

### Automatic (Recommended)
Simply push your changes to GitHub. The workflow defined in `.github/workflows/train.yml` will automatically trigger the Azure ML job using `train-job.yml`.

### Local Test (Optional)
To test the job submission locally (requires Azure CLI `az ml` extension):
```bash
az ml job create --file train-job.yml
```

## 📝 Configuration
- **Compute Target**: `gpu-cluster-t4` (Defined in `train-job.yml`)
- **Environment**: Built from `src/conda_env.yml`
- **Data**: Mounts `battery_storage` datastore
