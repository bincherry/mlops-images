import time
from typing import Dict, Optional, List
import logging
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from prometheus_client import Summary, Counter

# Initialize Prometheus metrics
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")
ERROR_COUNT = Counter("error_count", "Count of errors encountered in the system")

# Logger setup
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)

app = FastAPI()

@serve.deployment(name="EnhancedVLLMDeployment")
@serve.ingress(app)
class EnhancedVLLMDeployment:
    def __init__(
        self,
        engine_args: Dict,
        response_role: str,
        supported_models: List[str],
        default_model: str,
    ):
        logger.info(f"Initializing deployment with engine args: {engine_args}")
        self.response_role = response_role
        self.supported_models = supported_models
        self.default_model = default_model
        self.current_model = default_model
        self.models = {}
        self._initialize_models(engine_args)

    def _initialize_models(self, engine_args: Dict):
        """Initialize models specified in the supported_models list."""
        for model_name in self.supported_models:
            try:
                logger.info(f"Loading model: {model_name}")
                self.models[model_name] = AsyncLLMEngine.from_engine_args(
                    {**engine_args, "model": model_name}
                )
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                ERROR_COUNT.inc()
        if self.default_model not in self.models:
            raise ValueError("Default model is not in the supported models.")

    def switch_model(self, model_name: str):
        """Switch the current model at runtime."""
        if model_name not in self.models:
            raise HTTPException(status_code=400, detail="Unsupported model")
        self.current_model = model_name
        logger.info(f"Switched to model: {model_name}")

    @app.post("/v1/chat/completions")
    @REQUEST_TIME.time()
    async def create_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completions."""
        try:
            engine = self.models[self.current_model]
            model_config = await engine.get_model_config()
            openai_serving_chat = OpenAIServingChat(
                engine,
                model_config,
                served_model_names=[self.current_model],
                response_role=self.response_role,
            )

            generator = await openai_serving_chat.create_chat_completion(request)
            if isinstance(generator, ErrorResponse):
                ERROR_COUNT.inc()
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.code
                )
            elif request.stream:
                return StreamingResponse(
                    content=generator, media_type="text/event-stream"
                )
            else:
                assert isinstance(generator, ChatCompletionResponse)
                return JSONResponse(content=generator.model_dump())

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            ERROR_COUNT.inc()
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/v1/models")
    def list_models(self):
        """List available models."""
        return {"models": self.supported_models, "current_model": self.current_model}

    @app.get("/v1/models/{model_name}/health")
    def check_model_health(self, model_name: str):
        """Check the health of a specific model."""
        if model_name not in self.models:
            raise HTTPException(status_code=400, detail="Model not found")
        try:
            # Perform a lightweight health check
            engine = self.models[model_name]
            _ = engine.get_model_config()
            return {"model": model_name, "status": "healthy"}
        except Exception as e:
            logger.error(f"Health check failed for model {model_name}: {e}")
            ERROR_COUNT.inc()
            return {"model": model_name, "status": "unhealthy"}

    @app.post("/v1/models/{model_name}/switch")
    def switch_model_endpoint(self, model_name: str):
        """Endpoint to switch models."""
        self.switch_model(model_name)
        return {"message": f"Switched to model {model_name}"}


# Example usage of the enhanced deployment
def build_enhanced_app(cli_args: Dict):
    """Build the enhanced deployment."""
    supported_models = cli_args.get("supported_models", ["model-a", "model-b"])
    default_model = cli_args.get("default_model", supported_models[0])
    engine_args = {"tensor_parallel_size": cli_args.get("tensor_parallel_size", 1)}
    return EnhancedVLLMDeployment.bind(engine_args, "assistant", supported_models, default_model)


model = build_enhanced_app(
    {
        "supported_models": ["t5-small", "t5-large"],
        "default_model": "t5-small",
        "tensor_parallel_size": 2,
    }
)
