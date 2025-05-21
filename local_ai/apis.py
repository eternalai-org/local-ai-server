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
import uuid
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import multiprocessing
import sys

# Import schemas from schema.py (assumed to exist in your project)
from local_ai.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse
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

app = FastAPI(
    title="Local AI API",
    description="API for local AI model inference with load balancing",
    version="0.0.1",
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
MAX_RETRIES = 5  # Increased retries for better reliability
RETRY_DELAY = 1.0  # Increased delay between retries
HEALTH_CHECK_INTERVAL = 15  # More frequent health checks
MAX_RESPONSE_TIME_WINDOW = 200  # Larger window for better metrics
MAX_QUEUE_SIZE = 1000  # Maximum number of queued requests
MAX_WORKERS = min(multiprocessing.cpu_count() * 4, 32)  # Scale workers with CPU cores
BACKOFF_FACTOR = 1.5  # Exponential backoff factor for retries

class BackendInstance(BaseModel):
    """Model for a backend server instance"""
    instance_id: str
    port: int
    healthy: bool = True
    last_checked: float = Field(default_factory=time.time)
    error_count: int = 0
    response_times: List[float] = Field(default_factory=list)
    is_processing: bool = False
    _last_status_check: float = 0
    _status_cache_ttl: float = 1.0  # Cache status for 1 second
    
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
            
    async def check_processing_status(self, client: httpx.AsyncClient) -> bool:
        """Check if the instance is currently processing a request with caching"""
        current_time = time.time()
        if current_time - self._last_status_check < self._status_cache_ttl:
            return self.is_processing

        try:
            response = await client.get(f"http://localhost:{self.port}/slots", timeout=5.0)
            if response.status_code == 200:
                slots_data = response.json()
                self.is_processing = any(slot.get("is_processing", False) for slot in slots_data)
                self._last_status_check = current_time
                return self.is_processing
            else:
                logger.warning(f"Failed to get slots data from instance {self.instance_id}: {response.status_code}")
                return self.is_processing
        except Exception as e:
            logger.warning(f"Error checking processing status for instance {self.instance_id}: {str(e)}")
            return self.is_processing

class LoadBalancer:
    """Load balancer to distribute requests across multiple LLM server instances"""
    def __init__(self, health_check_interval: int = HEALTH_CHECK_INTERVAL):
        self.instances: Dict[str, BackendInstance] = {}
        self.health_check_interval = health_check_interval
        self.lock = asyncio.Lock()
        self.health_check_task = None
        self._instance_cache: Dict[str, float] = {}
        self._cache_ttl: float = 0.1
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.queue_workers: list = []
        self._shutdown_event = asyncio.Event()
        self._worker_count = MAX_WORKERS
        self._active_requests = 0
        self._max_concurrent_requests = POOL_CONNECTIONS
        self._request_semaphore = asyncio.Semaphore(self._max_concurrent_requests)
        self._backoff_times: Dict[str, float] = {}
    
    def update_instances(self, service_metadata: Dict[str, Any]):
        """Update instances from service metadata"""
        instances = service_metadata.get("instances", [])
        new_instances = {}
        
        for instance_info in instances:
            instance_id = instance_info.get("instance_id")
            port = instance_info.get("port")
            
            if instance_id and port:
                if instance_id in self.instances:
                    new_instances[instance_id] = self.instances[instance_id]
                else:
                    new_instances[instance_id] = BackendInstance(
                        instance_id=instance_id,
                        port=port
                    )
        
        self.instances = new_instances
        logger.info(f"Updated instances: {len(self.instances)} instances available")
    
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
        """Check health of a single instance with cooldown for unhealthy instances"""
        current_time = time.time()
        if not instance.healthy and current_time - instance.last_checked < 60:
            return  # Skip check if unhealthy and checked within last 60 seconds
        
        try:
            start_time = time.time()
            response = await client.get(
                f"http://localhost:{instance.port}/health",
                timeout=15.0
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                async with self.lock:
                    instance.record_success()
                    instance.record_response_time(duration)
                    instance.last_checked = current_time
                await instance.check_processing_status(client)
            else:
                logger.warning(f"Health check failed for instance {instance.instance_id}: Status code {response.status_code}")
                async with self.lock:
                    instance.record_error()
                    instance.last_checked = current_time
        except Exception as e:
            logger.warning(f"Health check failed for instance {instance.instance_id}: {str(e)}")
            async with self.lock:
                instance.record_error()
                instance.last_checked = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the load balancer instances"""
        healthy_instances = [i for i in self.instances.values() if i.healthy]
        idle_instances = [i for i in healthy_instances if not i.is_processing]
        
        return {
            "instances": [
                {
                    "instance_id": instance.instance_id,
                    "port": instance.port,
                    "healthy": instance.healthy,
                    "error_count": instance.error_count,
                    "avg_response_time": instance.avg_response_time(),
                    "is_processing": instance.is_processing,
                    "last_checked": instance.last_checked
                } for instance in self.instances.values()
            ],
            "healthy_count": len(healthy_instances),
            "idle_count": len(idle_instances),
            "processing_count": len(healthy_instances) - len(idle_instances),
            "total_count": len(self.instances)
        }
    
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
            
            idle_instances = [i for i in available_instances if not i.is_processing]
            
            if idle_instances:
                # Calculate weights based on response time and error count
                weights = []
                for instance in idle_instances:
                    response_time = instance.avg_response_time()
                    error_factor = 1.0 / (1.0 + instance.error_count)
                    weight = error_factor / (response_time + 1.0)
                    weights.append(weight)
                
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                    selected = random.choices(idle_instances, weights=normalized_weights, k=1)[0]
                else:
                    selected = random.choice(idle_instances)
            else:
                # If no idle instances, select based on response time and last checked
                selected = min(
                    available_instances,
                    key=lambda i: (i.avg_response_time(), i.last_checked)
                )
            
            # Update cache and last checked time
            self._instance_cache[selected.instance_id] = current_time
            selected.last_checked = current_time
            
            return selected
    
    async def refresh_processing_status(self, client: httpx.AsyncClient):
        """Refresh the processing status of all healthy instances with rate limiting"""
        healthy_instances = [i for i in self.instances.values() if i.healthy]
        semaphore_value = min(10, len(healthy_instances))  # Limit to 10 concurrent checks
        semaphore = asyncio.Semaphore(semaphore_value)
        
        async def check_with_semaphore(instance):
            async with semaphore:
                return await instance.check_processing_status(client)
        
        check_tasks = [check_with_semaphore(instance) for instance in healthy_instances]
        await asyncio.gather(*check_tasks, return_exceptions=True)
    
    def num_healthy_instances(self) -> int:
        return len([i for i in self.instances.values() if i.healthy])

    def num_processing_instances(self) -> int:
        return len([i for i in self.instances.values() if i.healthy and i.is_processing])

    async def start_queue_workers(self, client: httpx.AsyncClient):
        self._shutdown_event.clear()
        for _ in range(self._worker_count):
            worker = asyncio.create_task(self._queue_worker(client))
            self.queue_workers.append(worker)

    async def stop_queue_workers(self):
        self._shutdown_event.set()
        for worker in self.queue_workers:
            worker.cancel()
        await asyncio.gather(*self.queue_workers, return_exceptions=True)
        self.queue_workers.clear()

    async def _queue_worker(self, client: httpx.AsyncClient):
        while not self._shutdown_event.is_set():
            try:
                item = await self.request_queue.get()
                if item is None:
                    continue
                (endpoint, method, data, retries, future) = item
                try:
                    result = await self._execute_request_internal(client, endpoint, method, data, retries)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.request_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue worker error: {e}")

    async def execute_request(
        self, 
        client: httpx.AsyncClient,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
        retries: int = MAX_RETRIES
    ) -> Tuple[Dict, BackendInstance]:
        """Execute a request with improved concurrency control and backoff"""
        async with self._request_semaphore:
            self._active_requests += 1
            try:
                # Check if we need to queue
                healthy_count = self.num_healthy_instances()
                processing_count = self.num_processing_instances()
                
                if healthy_count > 0 and processing_count >= healthy_count:
                    if self.request_queue.qsize() >= MAX_QUEUE_SIZE:
                        raise HTTPException(
                            status_code=503,
                            detail="Service temporarily overloaded. Please try again later."
                        )
                    
                    # Queue the request with exponential backoff
                    loop = asyncio.get_event_loop()
                    future = loop.create_future()
                    await self.request_queue.put((endpoint, method, data, retries, future))
                    
                    try:
                        result = await asyncio.wait_for(future, timeout=HTTP_TIMEOUT)
                        return result
                    except asyncio.TimeoutError:
                        raise HTTPException(
                            status_code=504,
                            detail="Request timed out while waiting in queue"
                        )
                else:
                    return await self._execute_request_internal(client, endpoint, method, data, retries)
            finally:
                self._active_requests -= 1

    async def _execute_request_internal(
        self, 
        client: httpx.AsyncClient,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
        retries: int = MAX_RETRIES
    ) -> Tuple[Dict, BackendInstance]:
        """Internal method with improved error handling and backoff strategy"""
        if not self.instances:
            raise HTTPException(
                status_code=503,
                detail="No backend instances available"
            )

        tried_instances: Set[str] = set()
        last_error = None
        selected_instance = None

        for attempt in range(retries):
            try:
                instance = await self.get_next_instance()
                if not instance or instance.instance_id in tried_instances:
                    continue

                # Apply backoff if instance had recent errors
                if instance.instance_id in self._backoff_times:
                    backoff_time = self._backoff_times[instance.instance_id]
                    if time.time() < backoff_time:
                        continue
                    else:
                        del self._backoff_times[instance.instance_id]

                tried_instances.add(instance.instance_id)
                selected_instance = instance
                url = f"http://localhost:{instance.port}{endpoint}"

                async with self.lock:
                    instance.is_processing = True

                start_time = time.time()
                try:
                    if method.upper() == "POST":
                        response = await client.post(url, json=data, timeout=HTTP_TIMEOUT)
                    else:
                        response = await client.get(url, timeout=HTTP_TIMEOUT)

                    duration = time.time() - start_time
                    async with self.lock:
                        instance.record_success()
                        instance.record_response_time(duration)

                    if response.status_code >= 400:
                        error_text = response.text
                        if response.status_code >= 500:
                            async with self.lock:
                                instance.record_error()
                                instance.is_processing = False
                                # Apply exponential backoff
                                backoff_time = time.time() + (RETRY_DELAY * (BACKOFF_FACTOR ** attempt))
                                self._backoff_times[instance.instance_id] = backoff_time
                            continue

                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Backend request failed: {error_text}"
                        )

                    result = response.json()

                    if data and isinstance(data, dict) and not data.get("stream", False):
                        async with self.lock:
                            instance.is_processing = False

                    return result, instance

                except (httpx.TimeoutException, httpx.RequestError) as e:
                    logger.warning(f"Request failed for {url}: {str(e)}")
                    last_error = e
                    async with self.lock:
                        instance.record_error()
                        instance.is_processing = False
                        # Apply exponential backoff
                        backoff_time = time.time() + (RETRY_DELAY * (BACKOFF_FACTOR ** attempt))
                        self._backoff_times[instance.instance_id] = backoff_time
                    continue

            except Exception as e:
                logger.error(f"Error in request execution: {str(e)}")
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(min(RETRY_DELAY * (BACKOFF_FACTOR ** attempt), 10.0))

        if selected_instance:
            async with self.lock:
                selected_instance.is_processing = False

        raise HTTPException(
            status_code=503,
            detail=f"All requests failed after {retries} attempts. Last error: {str(last_error)}"
        )

# Initialize load balancer
load_balancer = LoadBalancer()

@app.on_event("startup")
async def startup_event():
    """Startup event handler: initialize the HTTP client and start health checks and queue workers"""
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
    await load_balancer.start_queue_workers(app.state.client)
    logger.info("Service started successfully with HTTP/2 support and queue workers")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down"""
    await load_balancer.stop_health_check()
    await load_balancer.stop_queue_workers()
    await app.state.client.aclose()
    logger.info("Service shutdown complete")

@app.get("/health")
@app.get("/v1/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok"
    }

@app.post("/update")
async def update(request: dict):
    """Update the service information and load balancer instances"""
    app.state.service_info = request
    load_balancer.update_instances(request)
    return {"status": "ok", "message": "Service info updated successfully"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completion endpoint with load balancing"""
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    
    # Remove None values for cleaner request
    request_dict = {k: v for k, v in request_dict.items() if v is not None}
    
    # Determine if we need to handle streaming with tools specially
    stream_with_tools = request.stream and request.tools
    
    if request.stream:
        if stream_with_tools:
            # For streaming with tools, we need to get a complete response first
            # and then simulate streaming
            request_dict["stream"] = False  # Temporarily disable streaming
            response_data, instance = await load_balancer.execute_request(
                app.state.client,
                "/v1/chat/completions",
                "POST",
                request_dict
            )
            
            # Format response to match OpenAI API schema
            formatted_response = ChatCompletionResponse(
                id=response_data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                object="chat.completion",
                created=response_data.get("created", int(time.time())),
                model=request.model,
                choices=response_data.get("choices", [])
            )
            
            async def fake_stream():
                try:
                    # Base structure for each chunk
                    base_chunk = {
                        "id": formatted_response.id,
                        "object": "chat.completion.chunk",
                        "created": formatted_response.created,
                        "model": formatted_response.model,
                    }
                    
                    # Add system_fingerprint only if it exists in the response
                    if hasattr(formatted_response, "system_fingerprint"):
                        base_chunk["system_fingerprint"] = formatted_response.system_fingerprint

                    choices = formatted_response.choices
                    if not choices:
                        # If no choices, return empty response and DONE
                        yield f"data: {json.dumps({**base_chunk, 'choices': []})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Step 1: Initial chunk with role for all choices
                    initial_choices = [
                        {
                            "index": choice.index,
                            "delta": {"role": "assistant", "content": ""},
                            "logprobs": None,
                            "finish_reason": None
                        }
                        for choice in choices
                    ]
                    yield f"data: {json.dumps({**base_chunk, 'choices': initial_choices})}\n\n"

                    # Step 2: Chunk with content or tool_calls for all choices
                    content_choices = []
                    for choice in choices:
                        message = choice.message
                        delta = {}
                        
                        # For tool calls responses
                        if message.tool_calls:
                            delta["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]
                            delta["reasoning_content"] = None
                        # For content responses (including vision content)
                        elif message.content:
                            # Handle both string and list content (vision content)
                            if isinstance(message.content, list):
                                # For vision content, we need to preserve the list structure
                                delta["content"] = [item.model_dump() for item in message.content]
                            else:
                                delta["content"] = message.content
                            delta["reasoning_content"] = None
                        else:
                            # Empty content/null case
                            delta["reasoning_content"] = None
                            
                        if delta:  # Only include choices with content
                            content_choices.append({
                                "index": choice.index,
                                "delta": delta,
                                "logprobs": None,
                                "finish_reason": None
                            })
                            
                    if content_choices:
                        yield f"data: {json.dumps({**base_chunk, 'choices': content_choices})}\n\n"

                    # Step 3: Final chunk with finish reason for all choices
                    finish_choices = [
                        {
                            "index": choice.index,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": choice.finish_reason
                        }
                        for choice in choices
                    ]
                    yield f"data: {json.dumps({**base_chunk, 'choices': finish_choices})}\n\n"

                    # Step 4: End of stream
                    yield "data: [DONE]\n\n"
                finally:
                    # Step 5: When streaming is done, update the instance status
                    if instance:
                        async with load_balancer.lock:
                            instance.is_processing = False
                            logger.info(f"Marked instance {instance.instance_id} as available after streaming")
            
            return StreamingResponse(
                fake_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering if used
                    "Content-Type": "text/event-stream",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            # For streaming without tools, use the stream generator
            instance = await load_balancer.get_next_instance()
            if not instance:
                raise RuntimeError("No backend instances available")
            
            async with load_balancer.lock:
                instance.is_processing = True
                logger.info(f"Marked instance {instance.instance_id} as busy for streaming request")
            
            async def stream_generator():
                try:
                    buffer = ""
                    async with app.state.client.stream(
                        "POST",
                        f"http://localhost:{instance.port}/v1/chat/completions",
                        json=request_dict,
                        timeout=STREAM_TIMEOUT,
                        headers={
                            "Accept": "application/json",
                            "X-Streaming-Request": "true"
                        }
                    ) as response:
                        if response.status_code != 200:
                            error_text = await response.text()
                            error_msg = f"data: {{\"error\":{{\"message\":\"{error_text}\",\"code\":{response.status_code}}}}}\n\n"
                            logger.error(f"Streaming error: {response.status_code} - {error_text}")
                            yield error_msg
                            return
                        
                        async for chunk in response.aiter_bytes():
                            buffer += chunk.decode('utf-8')
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                if line.strip():
                                    yield f"{line}\n\n"
                        
                        # Process any remaining data in the buffer
                        if buffer.strip():
                            yield f"{buffer}\n\n"
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"
                finally:
                    # Always mark the instance as available when done
                    async with load_balancer.lock:
                        instance.is_processing = False
                        logger.info(f"Marked instance {instance.instance_id} as available after streaming")
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering if used
                    "Content-Type": "text/event-stream",
                    "Transfer-Encoding": "chunked"
                }
            )
    else:
        # Non-streaming request
        logger.info("Non-streaming request")
        response_data, instance = await load_balancer.execute_request(
            app.state.client,
            "/v1/chat/completions",
            "POST",
            request_dict
        )
        return ChatCompletionResponse(
            id=response_data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            object="chat.completion",
            created=response_data.get("created", int(time.time())),
            model=request.model,
            choices=response_data.get("choices", [])
        )

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Embeddings endpoint with load balancing"""
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    response_data, _ = await load_balancer.execute_request(
        app.state.client,
        "/v1/embeddings",
        "POST",
        request_dict
    )
    return EmbeddingResponse(
        object=response_data.get("object", "list"),
        data=response_data.get("data", []),
        model=request.model
    )

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response