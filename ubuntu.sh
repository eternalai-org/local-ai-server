#!/bin/bash
set -o pipefail

# Logging functions
log_message() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --message \"$message\""
    fi
}

log_error() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --error \"$message\"" >&2
    fi
}

# Step 1: Search python package in /user
PYTHON_CMD=$(which python3)

# Step 2: Install all required packages at once
log_message "Installing required packages..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y pigz cmake libcurl4-openssl-dev
elif command -v yum &> /dev/null; then
    sudo yum install -y pigz cmake libcurl-openssl-dev
elif command -v dnf &> /dev/null; then
    sudo dnf install -y pigz cmake libcurl-openssl-dev
else
    log_error "No supported package manager found (apt-get, yum, or dnf)"
    exit 1
fi
log_message "All required packages installed successfully"

# Step 3: Check Docker installation
if ! command -v docker &> /dev/null; then
    log_message "Docker is not installed, installing..."
    sudo apt-get install -y docker.io
else
    log_message "Docker is already installed"
fi

# Step 4: Check NVIDIA Container Toolkit
if ! command -v nvidia-container-toolkit &> /dev/null; then
    log_message "NVIDIA Container Toolkit is not installed, installing..."
    # Add the package repositories
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Install the toolkit
    sudo apt update
    sudo apt install -y nvidia-container-toolkit

    # Configure Docker to use the NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker

    # Restart Docker
    sudo systemctl restart docker
else
    log_message "NVIDIA Container Toolkit is already installed"
fi

# Step 5: Pull llama-server cuda image
log_message "Pulling llama-server cuda image..."
docker pull ghcr.io/ggerganov/llama.cpp:server-cuda

# Step 6: Create and activate virtual environment
log_message "Creating virtual environment 'local_ai'..."
"$PYTHON_CMD" -m venv local_ai || handle_error $? "Failed to create virtual environment"

log_message "Activating virtual environment..."
source local_ai/bin/activate || handle_error $? "Failed to activate virtual environment"
log_message "Virtual environment activated."

# Step 7: Install local-ai toolkit
log_message "Setting up local-ai toolkit..."
if pip show local-ai &>/dev/null; then
    log_message "local-ai is installed. Checking for updates..."
    
    # Get installed version
    INSTALLED_VERSION=$(pip show local-ai | grep Version | awk '{print $2}')
    log_message "Current version: $INSTALLED_VERSION"
    
    # Get remote version (from GitHub repository without installing)
    log_message "Checking latest version from repository..."
    TEMP_VERSION_FILE=$(mktemp)
    if curl -s https://raw.githubusercontent.com/eternalai-org/local-ai-server/main/local_ai/__init__.py | grep -o "__version__ = \"[0-9.]*\"" | cut -d'"' -f2 > "$TEMP_VERSION_FILE"; then
        REMOTE_VERSION=$(cat "$TEMP_VERSION_FILE")
        rm "$TEMP_VERSION_FILE"
        
        log_message "Latest version: $REMOTE_VERSION"
        
         if [ "$(printf '%s\n' "$INSTALLED_VERSION" "$REMOTE_VERSION" | sort -V | head -n1)" = "$INSTALLED_VERSION" ] && [ "$INSTALLED_VERSION" != "$REMOTE_VERSION" ]; then
            log_message "New version available. Updating..."
            pip uninstall local-ai -y || handle_error $? "Failed to uninstall local-ai"
            pip install -q git+https://github.com/eternalai-org/local-ai-server.git || handle_error $? "Failed to update local-ai toolkit"
            log_message "local-ai toolkit updated to version $REMOTE_VERSION."
        else
            log_message "Already running the latest version. No update needed."
        fi
    else
        log_message "Could not check latest version. Proceeding with update to be safe..."
        pip uninstall local-ai -y || handle_error $? "Failed to uninstall local-ai"
        pip install -q git+https://github.com/eternalai-org/local-ai-server.git || handle_error $? "Failed to update local-ai toolkit"
        log_message "local-ai toolkit updated."
    fi
else
    log_message "Installing local-ai toolkit..."
    pip install -q git+https://github.com/eternalai-org/local-ai-server.git || handle_error $? "Failed to install local-ai toolkit"
    log_message "local-ai toolkit installed."
fi

log_message "Setup completed successfully."
