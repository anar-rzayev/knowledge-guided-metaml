#!/bin/bash

# Function to check dependencies and setup environment
setup_environment() {
    local in_container=$1
    
    if [[ $in_container == "true" ]]; then
        echo "Running in container environment..."
        command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not found"
    else
        echo "Setting up local environment..."
        if [ -f "requirements.txt" ]; then
            python3 -m pip install -r requirements.txt
        else
            echo "Error: requirements.txt not found!"
            exit 1
        fi
    fi
}

# Function to setup project
setup_project() {
    local project_dir="./meta/npt"
    echo "Installing project from: $project_dir"
    
    if [ -d "$project_dir" ]; then
        (cd "$project_dir" && pip install -e .) || {
            echo "Failed to install project"
            exit 1
        }
    else
        echo "Error: Project directory not found!"
        exit 1
    fi
}

# Set environment variables
export PYTHONHASHSEED=0
export QT_QPA_PLATFORM=xcb
export EXPERIMENT_TAG="V5_Single_0shot"

# Main execution
echo "Current working directory: $(pwd)"

# Check if running in Docker
setup_environment "${AM_I_IN_A_DOCKER_CONTAINER:-false}"

# Setup project
setup_project

# Run experiment
echo "Running experiment with tag: $EXPERIMENT_TAG"
HYDRA_FULL_ERROR=1 python run_metasupervised.py \
    +experiment/metasupervised=gym \
    logging.tags="[$EXPERIMENT_TAG]" \
    logging.type=terminal

echo "Experiment completed"