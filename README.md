# Local AI Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Deploy and manage large language models locally with minimal setup.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🔭 Overview

The **Local AI Toolkit** empowers developers to deploy state-of-the-art large language models directly on their machines. It provides an OpenAI-compatible API interface, allowing you to seamlessly integrate local models into your applications while maintaining privacy and reducing costs.

## ✨ Features

- **Simple Deployment**: Get models running with minimal configuration
- **Filecoin Integration**: Upload and download models directly from decentralized storage
- **OpenAI-Compatible API**: Seamless integration with existing applications
- **Automatic Model Management**: Download, cache, and manage models efficiently
- **Health Monitoring**: Auto-recovery and health checks for reliable operation
- **Vision Model Support**: Run both text and vision models locally
- **Idle Timeout**: Automatic shutdown after 10 minutes of inactivity

## 📦 Installation

### MacOS

```bash
bash mac.sh
```

### Verification

Confirm successful installation:

```bash
source local_ai/bin/activate
local-ai --version
```

## 🚀 Usage

### Managing Models

```bash
# Check if model is available locally
local-ai check --hash <filecoin_hash>

# Upload a model to Filecoin
local-ai upload --folder-name <folder_name> --task <task>

# Download a model from Filecoin
local-ai download --hash <filecoin_hash>

# Start a model
local-ai start --hash <filecoin_hash>

# Example
local-ai start --hash bafkreiecx5ojce2tceibd74e2koniii3iweavknfnjdfqs6ows2ikoow6m

# Check running models
local-ai status

# Stop the current model
local-ai stop
```

### Start Options
- `--port`: Service port (default: 8080)
- `--host`: Host address (default: "0.0.0.0")
- `--context-length`: Model context length (default: 32768)

### Important Notes on Uploading Models

When using the `upload` command, the following flags are required:

- **`--folder-name`**: Specifies the directory containing your model. The model file must have the same name as this folder.

**Example Upload Command:**

```bash
# Upload a BERT model for text classification
local-ai upload --folder-name bert-classifier
```

Make sure your folder structure follows this convention:
```
llama2-7b/
├── llama2-7b (the model file with `gguf` format)
```

## 📡 API Reference

### Chat Completion
```python
POST /v1/chat/completions
{
    "model": "local-model",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,  # optional
    "max_tokens": 4096   # optional
}
```

### Embeddings
```python
POST /v1/embeddings
{
    "model": "local-model",
    "input": ["Hello, how are you?"]
}
```

## 📝 Logs
- `logs/api.log`: API service logs
- `logs/ai.log`: AI service logs

## 👥 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 🔧 Requirements

- Python >= 3.9
- See `setup.py` for full list of dependencies