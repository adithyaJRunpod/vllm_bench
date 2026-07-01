from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
import json
import logging
import os
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from utils import format_chat_prompt, create_error_response
from models import GenerationRequest, GenerationResponse, ChatCompletionRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize the vLLM engine on startup and cleanup on shutdown"""
    await create_engine()
    yield
    global engine, engine_ready
    if engine:
        logger.info("Shutting down vLLM engine...")
        engine = None
        engine_ready = False
        logger.info("vLLM engine shutdown complete")


app = FastAPI(title="vLLM Load Balancing Server", version="1.0.0", lifespan=lifespan)

engine: Optional[AsyncLLMEngine] = None
engine_ready = False


async def create_engine():
    """Initialize the vLLM engine"""
    global engine, engine_ready

    try:
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")

        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            dtype=os.getenv("DTYPE", "auto"),
            trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true",
            max_model_len=int(os.getenv("MAX_MODEL_LEN")) if os.getenv("MAX_MODEL_LEN") else None,
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95")),
            enforce_eager=os.getenv("ENFORCE_EAGER", "false").lower() == "true",
            enable_prefix_caching=os.getenv("ENABLE_PREFIX_CACHING", "false").lower() == "true",
        )

        engine = AsyncLLMEngine.from_engine_args(engine_args)
        engine_ready = True
        logger.info(f"vLLM engine initialized successfully with model: {model_name}")

    except Exception as e:
        logger.error(f"Failed to initialize vLLM engine: {str(e)}")
        engine_ready = False
        raise


@app.get("/ping")
async def health_check():
    """Health check endpoint required by RunPod load balancer"""
    if not engine_ready:
        logger.debug("Health check: Engine initializing")
        return JSONResponse(
            content={"status": "initializing"},
            status_code=status.HTTP_204_NO_CONTENT
        )

    logger.debug("Health check: Engine healthy")
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "vLLM Load Balancing Server",
        "status": "ready" if engine_ready else "initializing",
        "endpoints": {
            "health": "/ping",
            "generate": "/v1/completions",
            "chat": "/v1/chat/completions"
        }
    }


@app.post("/v1/completions", response_model=GenerationResponse)
async def generate_completion(request: GenerationRequest):
    """Generate text completion"""
    logger.info(f"Received completion request: max_tokens={request.max_tokens}, temperature={request.temperature}, stream={request.stream}")

    if not engine_ready or engine is None:
        logger.warning("Completion request rejected: Engine not ready")
        error_response = create_error_response("ServiceUnavailable", "Engine not ready")
        raise HTTPException(status_code=503, detail=error_response.model_dump())

    try:
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
        )

        request_id = random_uuid()

        if request.stream:
            return StreamingResponse(
                stream_completion(request.prompt, sampling_params, request_id),
                media_type="text/event-stream",
            )
        else:
            results = engine.generate(request.prompt, sampling_params, request_id)
            final_output = None
            async for output in results:
                final_output = output

            if final_output is None:
                request_id = random_uuid()
                error_response = create_error_response("GenerationError", "No output generated", request_id)
                raise HTTPException(status_code=500, detail=error_response.model_dump())

            generated_text = final_output.outputs[0].text
            finish_reason = final_output.outputs[0].finish_reason

            if hasattr(final_output, 'prompt_token_ids') and final_output.prompt_token_ids is not None:
                prompt_tokens = len(final_output.prompt_token_ids)
            else:
                prompt_tokens = len(request.prompt.split())

            completion_tokens = len(final_output.outputs[0].token_ids)

            logger.info(f"Completion generated: {completion_tokens} tokens, finish_reason={finish_reason}")
            return GenerationResponse(
                text=generated_text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )

    except HTTPException:
        raise
    except Exception as e:
        request_id = random_uuid()
        logger.error(f"Generation failed (request_id={request_id}): {str(e)}", exc_info=True)
        error_response = create_error_response("GenerationError", f"Generation failed: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=error_response.model_dump())


async def stream_completion(prompt: str, sampling_params: SamplingParams, request_id: str) -> AsyncGenerator[str, None]:
    """Stream completion generator"""
    try:
        results = engine.generate(prompt, sampling_params, request_id)
        async for output in results:
            for output_item in output.outputs:
                yield f"data: {json.dumps({'text': output_item.text, 'finish_reason': output_item.finish_reason})}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    logger.info(f"Received chat completion request: {len(request.messages)} messages, max_tokens={request.max_tokens}, temperature={request.temperature}")

    if not engine_ready or engine is None:
        logger.warning("Chat completion request rejected: Engine not ready")
        error_response = create_error_response("ServiceUnavailable", "Engine not ready")
        raise HTTPException(status_code=503, detail=error_response.model_dump())

    try:
        messages = request.messages
        if not messages:
            error_response = create_error_response("ValidationError", "No messages provided")
            raise HTTPException(status_code=400, detail=error_response.model_dump())

        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
        prompt = format_chat_prompt(messages, model_name)

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

        request_id = random_uuid()
        results = engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for output in results:
            final_output = output

        if final_output is None:
            error_response = create_error_response("GenerationError", "No output generated", request_id)
            raise HTTPException(status_code=500, detail=error_response.model_dump())

        generated_text = final_output.outputs[0].text
        completion_tokens = len(final_output.outputs[0].token_ids)
        logger.info(f"Chat completion generated: {completion_tokens} tokens, finish_reason={final_output.outputs[0].finish_reason}")

        if hasattr(final_output, 'prompt_token_ids') and final_output.prompt_token_ids is not None:
            prompt_tokens = len(final_output.prompt_token_ids)
        else:
            prompt_tokens = len(prompt.split())

        return {
            "id": request_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": final_output.outputs[0].finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        request_id = random_uuid()
        logger.error(f"Chat completion failed (request_id={request_id}): {str(e)}", exc_info=True)
        error_response = create_error_response("ChatCompletionError", f"Chat completion failed: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=error_response.model_dump())


if __name__ == "__main__":
    port = int(os.getenv("PORT", 80))
    logger.info(f"Starting vLLM server on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
