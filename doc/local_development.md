# Local Development Setup (Without Docker)

If you prefer to run locally without Docker:

## Prerequisites

- **Python 3.8+** (3.11 recommended)
- **UV** - Fast Python package installer ([Install](https://github.com/astral-sh/uv): `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **DVC** - Installed automatically with dependencies
- **Git** & **Make** - Usually pre-installed
- **Dagshub account** - [Sign up](https://dagshub.com) for data versioning (optional)

> **Windows Users**: This project uses Makefiles and shell scripts that require a Unix-like environment. On Windows, please use one of the following:
> - **Git Bash** (recommended) - Comes with Git for Windows
> - **WSL (Windows Subsystem for Linux)** - Full Linux environment
> - **MSYS2/MinGW** - Unix-like environment for Windows
>
> The Makefile and shell commands will not work in native Windows PowerShell or CMD.

## Setup Steps

1. **Clone and setup Python**

```bash
git clone https://github.com/chrmei/MLOps_accidents.git
cd MLOps_accidents
# Using pyenv: pyenv install 3.11.0 && pyenv local 3.11.0
# Using asdf: asdf install python 3.11.0 && asdf reshim python
```

2. **Install dependencies**

```bash
make install-dev  # Installs project + dev dependencies (pytest, black, isort, mypy, etc.)
# Alternative: uv pip install -e ".[dev]"
```

3. **Verify installation**

```bash
python --version && make help
```

## DVC and Dagshub Setup (Optional)

DVC is used for data versioning. To set it up:

```bash
# Step 1: Create .env file from template (if .env.example exists)
cp .env.example .env

# Step 2: MANUALLY EDIT .env file with your Dagshub credentials:
#   - DAGSHUB_USERNAME: Your Dagshub username
#   - DAGSHUB_TOKEN: Get from https://dagshub.com/user/settings/tokens
#   - DAGSHUB_REPO: Your repository (e.g., chrmei/MLOps_accidents)
# 
# IMPORTANT: You must manually edit .env before running the next commands!

# Step 3: Initialize DVC and configure remote using Makefile
make dvc-init
make dvc-setup-remote
```

**Important Notes**:
- The `.env` file must be **manually edited** with your credentials before running `make dvc-setup-remote`
- The `.env` file is gitignored and will never be committed
- Each team member should create their own `.env` file with their personal Dagshub credentials

## Run the Pipeline Locally

```bash
# 1. Import raw data (downloads 4 CSV files from AWS S3)
make run-import

# 2. Preprocess data (creates train/test splits in data/preprocessed/)
make run-preprocess

# 3. Train models (saves to models/{model_type}_model.joblib)
make run-train

# 4. Make predictions
make run-predict                    # Interactive mode
make run-predict-file FILE=path     # From JSON file
```
