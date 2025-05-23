"""
This module provides a FastAPI application that acts as a proxy or processor for chat completion and embedding requests,
forwarding them to an underlying service running on a local port. It handles both text and vision-based chat completions,
as well as embedding generation, with support for streaming responses.
"""

import os
import logging
import httpx
import asyncio
import time
import json
import random
import uvicorn
from typing import Dict, List, Optional, Tuple, Any, Set
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import multiprocessing
import sys
from local_ai.config import CONFIG
from contextlib import asynccontextmanager


# Import schemas from schema.py (assumed to exist in your project)
from local_ai.schema import (
    ChatCompletionRequest
)

class ErrorHandlingStreamHandler(logging.StreamHandler):
    """Custom stream handler that handles I/O errors gracefully"""
    def emit(self, record):
        try:
            super().emit(record)
        except OSError as e:
            if e.errno == 5:  # Input/output error
                # Silently ignore I/O errors
                pass
            else:
                # Re-raise other OSErrors
                raise

# Set up logging with environment-based level and error handling
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING if os.getenv("ENV") == "production" else logging.INFO)

# Create and configure the error handling handler
handler = ErrorHandlingStreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Remove any existing handlers to avoid duplicate logging
for existing_handler in logger.handlers[:]:
    if not isinstance(existing_handler, ErrorHandlingStreamHandler):
        logger.removeHandler(existing_handler)

# Configure uvicorn access logger with error handling
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = []  # Remove existing handlers
uvicorn_access_handler = ErrorHandlingStreamHandler(sys.stderr)
uvicorn_access_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
uvicorn_access_logger.addHandler(uvicorn_access_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    limits = httpx.Limits(
        max_connections=POOL_CONNECTIONS,
        max_keepalive_connections=POOL_CONNECTIONS,
        keepalive_expiry=POOL_KEEPALIVE
    )
    app.state.client = httpx.AsyncClient(
        limits=limits,
        timeout=HTTP_TIMEOUT,
        transport=httpx.AsyncHTTPTransport(
            retries=MAX_RETRIES,
            verify=False
        ),
        http2=True
    )
    await load_balancer.start_health_check(app.state.client)
    logger.info("Service started successfully with HTTP/2 support")
    
    yield
    
    # Shutdown
    await load_balancer.stop_health_check()
    await app.state.client.aclose()
    logger.info("Service shutdown complete")

app = FastAPI(
    title="EternalAI Server",
    description="Server for AI model inference with load balancing",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Allow additional origins from environment variable
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend(os.getenv("ALLOWED_ORIGINS").split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Constants
POOL_CONNECTIONS = min(multiprocessing.cpu_count() * 200, 2000)  # Increased for better concurrency
POOL_KEEPALIVE = 120  # Increased keepalive time
HTTP_TIMEOUT = 60.0  # Increased timeout for large requests
STREAM_TIMEOUT = 600.0  # Increased for long-running streams
MAX_RETRIES = 1  # Increased retries for better reliability
RETRY_DELAY = 1.0  # Increased delay between retries
HEALTH_CHECK_INTERVAL = 15  # More frequent health checks
MAX_RESPONSE_TIME_WINDOW = 200  # Larger window for better metrics
MAX_QUEUE_SIZE = 1000  # Maximum number of queued requests
MAX_WORKERS = min(multiprocessing.cpu_count() * 4, 32)  # Scale workers with CPU cores
BACKOFF_FACTOR = 1.5  # Exponential backoff factor for retries

class BackendInstance(BaseModel):
    """Model for a backend server instance"""
    instance_id: str
    url: str
    healthy: bool = True
    last_checked: float = Field(default_factory=time.time)
    error_count: int = 0
    response_times: List[float] = Field(default_factory=list)
    is_processing: bool = False
    
    def avg_response_time(self) -> float:
        """Calculate average response time or return max value if no data"""
        if not self.response_times:
            return float('inf')
        return sum(self.response_times) / len(self.response_times)
    
    def record_response_time(self, duration: float):
        """Record response time and maintain a sliding window of last N responses"""
        self.response_times.append(duration)
        if len(self.response_times) > MAX_RESPONSE_TIME_WINDOW:
            self.response_times.pop(0)
    
    def record_error(self):
        """Record an error and potentially mark instance as unhealthy"""
        self.error_count += 1
        if self.error_count >= 3:  # Circuit breaker opens after 3 consecutive errors
            self.healthy = False
            logger.warning(f"Backend instance {self.instance_id} marked as unhealthy after {self.error_count} errors")
    
    def record_success(self):
        """Record a successful request and reset error count"""
        self.error_count = 0
        if not self.healthy:
            self.healthy = True
            logger.info(f"Backend instance {self.instance_id} is now healthy again")

class LoadBalancer:
    """Load balancer to distribute requests across multiple LLM server instances"""
    def __init__(self, health_check_interval: int = HEALTH_CHECK_INTERVAL):
        self.instances: Dict[str, BackendInstance] = {}
        self.health_check_interval = health_check_interval
        self.lock = asyncio.Lock()
        self.health_check_task = None
        self._instance_cache: Dict[str, float] = {}
        self._cache_ttl: float = 0.1
        
        # Initialize instances from CONFIG
        urls = CONFIG.get("urls", [])
        for i, url_info in enumerate(urls):
            # Get the URL string from the config
            url = url_info.get("url") if isinstance(url_info, dict) else url_info
            instance_id = f"instance_{i}"
            self.instances[instance_id] = BackendInstance(
                instance_id=instance_id,
                url=url
            )
        logger.info(f"Initialized {len(self.instances)} instances from config")
    
    async def start_health_check(self, client: httpx.AsyncClient):
        """Start the health check task"""
        self.health_check_task = asyncio.create_task(self._health_check_loop(client))
        
    async def stop_health_check(self):
        """Stop the health check task"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self, client: httpx.AsyncClient):
        """Periodically check health of all instances"""
        while True:
            try:
                await self._check_all_instances(client)
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_all_instances(self, client: httpx.AsyncClient):
        """Check health of all instances concurrently"""
        check_tasks = []
        for instance in self.instances.values():
            check_tasks.append(self._check_instance_health(client, instance))
        await asyncio.gather(*check_tasks, return_exceptions=True)
    
    async def _check_instance_health(self, client: httpx.AsyncClient, instance: BackendInstance):
        """Check health of a single instance directly"""
        current_time = time.time()
        if not instance.healthy and current_time - instance.last_checked < 60:
            return  # Skip check if unhealthy and checked within last 60 seconds
        
        try:
            start_time = time.time()
            response = await client.get(
                f"{instance.url}/health",
                timeout=15.0
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                instance.record_success()
                instance.record_response_time(duration)
                instance.last_checked = current_time
            else:
                instance.record_error()
                logger.warning(f"Health check failed for instance {instance.instance_id}: {response.status_code}")
        except Exception as e:
            instance.record_error()
            logger.error(f"Error checking health for instance {instance.instance_id}: {str(e)}")
    
    async def get_next_instance(self) -> Optional[BackendInstance]:
        """Get the next available instance using a weighted selection based on performance metrics"""
        async with self.lock:
            current_time = time.time()
            healthy_instances = [i for i in self.instances.values() if i.healthy]
            
            if not healthy_instances:
                logger.error("No healthy instances available!")
                if not self.instances:
                    return None
                return random.choice(list(self.instances.values()))
            
            # Filter out instances that were recently selected
            available_instances = [
                i for i in healthy_instances
                if i.instance_id not in self._instance_cache
                or current_time - self._instance_cache[i.instance_id] > self._cache_ttl
            ]
            
            if not available_instances:
                available_instances = healthy_instances
            
            # Calculate weights based on response time and error count
            weights = []
            for instance in available_instances:
                response_time = instance.avg_response_time()
                error_factor = 1.0 / (1.0 + instance.error_count)
                weight = error_factor / (response_time + 1.0)
                weights.append(weight)
            
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                selected = random.choices(available_instances, weights=normalized_weights, k=1)[0]
            else:
                selected = random.choice(available_instances)
            
            # Update cache and last checked time
            self._instance_cache[selected.instance_id] = current_time
            selected.last_checked = current_time
            
            return selected

    async def execute_request(
        self, 
        client: httpx.AsyncClient,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
        retries: int = MAX_RETRIES
    ) -> Tuple[Dict, BackendInstance]:
        """Execute a request directly to the selected instance"""
        instance = await self.get_next_instance()
        if not instance:
            raise HTTPException(status_code=503, detail="No healthy instances available")

        start_time = time.time()
        try:
            # Send request directly to the instance
            response = await client.request(
                method,
                f"{instance.url}{endpoint}",
                json=data,
                timeout=HTTP_TIMEOUT
            )
            
            if response.status_code >= 500:
                instance.record_error()
                if retries > 0:
                    await asyncio.sleep(RETRY_DELAY)
                    return await self.execute_request(client, endpoint, method, data, retries - 1)
                raise HTTPException(status_code=response.status_code, detail="Backend service error")
            
            instance.record_success()
            instance.record_response_time(time.time() - start_time)
            return response.json(), instance
            
        except Exception as e:
            instance.record_error()
            if retries > 0:
                await asyncio.sleep(RETRY_DELAY)
                return await self.execute_request(client, endpoint, method, data, retries - 1)
            raise HTTPException(status_code=500, detail=str(e))

# Initialize load balancer
load_balancer = LoadBalancer()

@app.get("/health")
@app.get("/v1/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok"
    }

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests"""
    async with httpx.AsyncClient() as client:
        try:
            print(request)
            dict_request = request.dict()
            dict_request["model"] = CONFIG["model"]["id"]
            print(dict_request)

            if request.stream:
                async def stream_generator():
                    try:
                        response, instance = await load_balancer.execute_request(
                            client,
                            "/v1/chat/completions",
                            data=dict_request
                        )
                        
                        async for chunk in response:
                            if isinstance(chunk, dict):
                                yield f"data: {json.dumps(chunk)}\n\n"
                            else:
                                yield f"data: {chunk}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"Error in stream: {str(e)}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            else:
                response, instance = await load_balancer.execute_request(
                    client,
                    "/v1/chat/completions",
                    data=dict_request
                )
                return response

        except Exception as e:
            logger.error(f"Error processing chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    uvicorn.run(
        "local_ai.apis:app",
        host="0.0.0.0",
        port=CONFIG.get("proxy_port", 65534),
        workers=CONFIG.get("workers", 1)
    )