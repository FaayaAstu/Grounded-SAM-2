#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_VERSION="3.10.12"
ENV_NAME="sam-dino2"

# --- Pre-requisites Check ---
if ! command -v pyenv &> /dev/null
then
    echo "Error: pyenv could not be found. Please install pyenv and pyenv-virtualenv."
    exit 1
fi

# --- Initialize pyenv for this script session ---
echo "Initializing pyenv for the script..."
export PYENV_ROOT="$HOME/.pyenv"
[[ -d "$PYENV_ROOT/bin" ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# --- Install System Dependencies ---
# Needed for pyenv Python builds and some package dependencies
echo "Installing system dependencies..."
apt-get update && \
    apt-get install -y --no-install-recommends git wget build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl llvm libncursesw5-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- pyenv Environment Setup ---
echo "Checking for Python version ${PYTHON_VERSION}..."
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "Installing Python ${PYTHON_VERSION} with pyenv..."
    pyenv install ${PYTHON_VERSION}
else
    echo "Python ${PYTHON_VERSION} already installed."
fi

echo "Checking for pyenv virtual environment '${ENV_NAME}'..."
# Construct the expected path to the virtual environment
VENV_PATH="$PYENV_ROOT/versions/${ENV_NAME}" # pyenv virtualenvs live directly under versions now
if ! pyenv virtualenvs --bare | grep -q "^${ENV_NAME}$"; then
    echo "Creating pyenv virtual environment '${ENV_NAME}' using Python ${PYTHON_VERSION}..."
    pyenv virtualenv ${PYTHON_VERSION} ${ENV_NAME}
else
    echo "Virtual environment '${ENV_NAME}' already exists."
fi

# --- Activate Environment by Sourcing ---
ACTIVATE_SCRIPT="${VENV_PATH}/bin/activate"
if [[ -f "$ACTIVATE_SCRIPT" ]]; then
    echo "Activating environment '${ENV_NAME}' by sourcing ${ACTIVATE_SCRIPT}..."
    source "${ACTIVATE_SCRIPT}"
else
    echo "Error: Activation script not found at ${ACTIVATE_SCRIPT}"
    # Attempt to find the correct path in case the structure changed slightly
    ALT_VENV_PATH=$(pyenv prefix ${ENV_NAME})
    ALT_ACTIVATE_SCRIPT="${ALT_VENV_PATH}/bin/activate"
    echo "Attempting to find activation script at alternate path: ${ALT_ACTIVATE_SCRIPT}"
    if [[ -f "$ALT_ACTIVATE_SCRIPT" ]]; then
         echo "Found activation script at alternate path: ${ALT_ACTIVATE_SCRIPT}. Sourcing..."
         source "${ALT_ACTIVATE_SCRIPT}"
    else
        echo "Error: Could not find activation script at standard or alternate path."
        exit 1
    fi
fi

# --- Verification Step ---
echo "--- Verifying Environment Activation ---"
echo "which python: $(which python)"
echo "python version: $(python --version)"
echo "which pip: $(which pip)"
echo "VIRTUAL_ENV variable: $VIRTUAL_ENV"
if [[ -z "$VIRTUAL_ENV" || ! "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
    echo "Error: Environment does not seem to be activated correctly after sourcing!"
    # Decide whether to exit or proceed cautiously
    # exit 1 # Uncomment this line to stop if activation failed
fi
echo "--- Verification End ---"

echo "Installing dependencies from requirements.txt (including PyTorch/GPU libraries)..."
python -m pip install -r requirements.txt

echo "Installing segment-anything-hq package..."
python -m pip install -e .

echo "Installing gdino"
pip install --no-build-isolation -e grounding_dino

echo "Creating pretrained checkpoint directory..."
mkdir -p pretrained_checkpoint

echo "Downloading sam models using downloader script..."
cd checkpoints && \
./download_ckpts.sh && \
cd ..
echo "Model check/download complete."

echo "Downloading gdino models using downloader script..."
cd gdino_checkpoints
./download_ckpts.sh
cd ..
echo "Grounding dino download complete"

# --- Deactivate (Optional but good practice in scripts) ---
echo "Deactivating environment..."
# Check if deactivate command exists before running it
#if command -v deactivate &> /dev/null; then
#    deactivate
#else
#    echo "Warning: 'deactivate' command not found. Skipping deactivation."
#fi

# --- Completion ---
echo ""
echo "Setup complete using pyenv."
echo "The environment '${ENV_NAME}' was set for this script session using 'pyenv shell'."
echo "For future interactive use, activate it with:"
echo "  pyenv activate ${ENV_NAME}"
echo "Or set it for the local directory:"
echo "  pyenv local ${ENV_NAME}"
echo "(Remember to have 'eval \"\$(pyenv init -)\"' and 'eval \"\$(pyenv virtualenv-init -)\"' in your shell configuration)"
