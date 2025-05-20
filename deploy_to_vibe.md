# Deploy Your Local Large Language Model to Vibe

This guide explains how to deploy your local large language model to the Vibe platform.

## Prerequisites

- Your local model must be in `gguf` format (compatible with `llama.cpp`)
- You need a Lighthouse account with an API key for uploading the model

## Step 1: Install the Local AI Tool

Install the `local-ai` tool required for uploading models to Lighthouse:

```bash
bash mac.sh
```

## Step 2: Activate the Virtual Environment

```bash
source local_ai/bin/activate
```

## Step 3: Upload Your Model to Lighthouse

### Prepare Your Model Files

The `local-ai` tool requires a specific folder structure for your model:

1. Create a folder with the same name as your model (without the file extension)
2. Place your model file inside this folder and rename it to match the folder name

Example:
- Original model file: `llama3-8b.gguf`
- Create folder: `llama3-8b`
- Place the model file inside the folder and rename it to `llama3-8b`

After doing this, your folder structure should look like this:

```bash
llama3-8b/
└── llama3-8b  # This was renamed from llama3-8b.gguf
```

### Upload the Model

```bash
local-ai upload --folder-name llama3-8b
```

### Model Metadata Options

You can specify additional metadata for your model:

| Parameter | Description |
|-----------|-------------|
| `ram` | Required RAM amount for 32k context window |
| `hf-repo` | HuggingFace repository ID of the model |
| `hf-file` | File name of the model in HuggingFace repo |

After uploading completes, the metadata will be saved in a file named `llama3-8b_metadata.json`. This file contains the Filecoin hash (IPFS CID) needed for deployment.

## Step 4: Deploy Your Model to Vibe

1. Go to the [Vibe Create Agent Website](https://staging.eternalai.org/for-developers/create?tab=5)
2. Create a new local model agent using the following information:

### Agent Configuration

| Field | Description |
|-------|-------------|
| Agent type | Large Language Model |
| Category | Model |
| Network | Base Sepolia |
| Agent avatar | Your model's avatar image |
| Agent name | Your model's name |
| Display name | Your model's display name |
| Short description | Brief description of your model |
| Description | Detailed description of your model |
| Model | `ipfs://[YOUR_FILECOIN_HASH]` (e.g., `ipfs://bafkreibl7efovdbo5vb3u6pcga2e2s7bqnyf7lrgaliaeldlaqtvgjmd4q`) |
| Dependent agents | Leave blank |
| Required free RAM | Estimated RAM needed for 32k context window |
| Required free Disk | Your model size in GB (double the actual size to account for decompression) |

## Troubleshooting

If you encounter issues during the upload or deployment process, check:
- Your API key is valid and has sufficient permissions
- The model file is in the correct format
- The folder structure follows the naming convention
- You have sufficient disk space and RAM for the model 