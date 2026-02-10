# serve_qwen_vllm.py
import ray
from ray import serve
import os
import asyncio
import json
from ollama import Client, AsyncClient
import asyncio
from fastapi import FastAPI,HTTPException
import sys

from mlc_llm import MLCEngine, AsyncMLCEngine
from uuid import uuid4
from fastapi.responses import StreamingResponse

# Initialize Ray if not already initialized.
# Using "auto" to connect to an existing Ray cluster or start a new one.
# For local development, `ray.init()` without arguments is often sufficient.
# if not ray.is_initialized():
#     ray.init(address="auto")

# Initialize Ray with explicit dashboard settings
if not ray.is_initialized():
    ray.init(
        include_dashboard=True,
        dashboard_host='127.0.0.1',
        dashboard_port=8265,
        _temp_dir='/tmp/ray'  # Explicit temp directory
    )
    print(f"Ray initialized. Dashboard should be available at http://127.0.0.1:8265")

app = FastAPI()

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class QwenOllamaDeployment:
    def __init__(self):
            
        self.message = {'role': 'user', 'content': 'Why is the sky blue?'}
        self.model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
        self.engine = AsyncMLCEngine(self.model,mode = "interactive" )

    async def stream_tokens(self, messages):
        try:
            # Sampling parameters for response generation

            # prompt = self.prepare_prompt(messages)
            request_id = str(uuid4())
            
            previous_text = ""

            async for response in await self.engine.chat.completions.create(
                messages=[{"role": "user", "content": messages}],
                model=self.model,
                stream=True,
                ):
                for choice in response.choices:
                    yield choice.delta.content

        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    @app.post("/generate")
    async def generate(self, payload: dict):
        
        prompt = payload.get("prompt")
        messages = [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        try:
            return StreamingResponse(
                self.stream_tokens(messages[0]['content']),
                media_type="text/event-stream"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
# Bind the deployment to an application
qwen_app = QwenOllamaDeployment.bind()

# 2: Deploy the application locally.
serve.run(qwen_app,route_prefix="/",
        name="qwen_deployment", blocking=True)