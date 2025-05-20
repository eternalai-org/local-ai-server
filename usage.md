# Local AI

A lightweight wrapper for running local Large Language Models with OpenAI-compatible API.

## Quick Start

1. **Setup**
   ```bash
   bash mac.sh
   source local_ai/bin/activate
   ```

2. **Start Service**
   ```bash
   local-ai start --hash <filecoin-hash>
   ```

3. **Test Connection**
   ```python
   import requests
   
   response = requests.post(
       "http://localhost:8080/v1/chat/completions",
       json={
           "model": "local-model",
           "messages": [{"role": "user", "content": "Hello!"}]
       }
   )
   print(response.json())
   ```

## API Reference

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

### Vision Chat Completion
```python
POST /v1/chat/completions
{
    "model": "local-model",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."  # base64 encoded image
                    }
                }
            ]
        }
    ]
}
```

### Embeddings
```python
POST /v1/embeddings
{
    "model": "local-model",
    "input": ["Hello, how are you?"]
}

# Multiple inputs example
POST /v1/embeddings
{
    "model": "local-model",
    "input": [
        "Hello, how are you?",
        "What's the weather like?",
        "Tell me a story"
    ]
}
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `start --hash <hash>` | Start Local AI service |
| `stop` | Stop running service |
| `status` | Check running model hash |
| `restart` | Restart current service |
| `download --hash <hash>` | Download model without starting |

### Start Options
- `--port`: Service port (default: 8080)
- `--host`: Host address (default: "0.0.0.0")
- `--context-length`: Model context length (default: 32768)

## Logs
- `logs/api.log`: API service logs
- `logs/ai.log`: Local AI service logs

## Features
- Automatic model downloading and caching
- OpenAI-compatible API
- Single model instance management
- Auto-recovery and health checks
- 10-minute idle timeout
- Support for text and vision models
