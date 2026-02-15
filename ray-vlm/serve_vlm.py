# serve_qwen_vllm.py
import os
import sys
import ray
import json
import torch

import numpy as np
from ray import serve
from PIL import Image
from uuid import uuid4
from typing import List
from typing import Annotated
from threading import Thread
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI,HTTPException,File, UploadFile,Form
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoTokenizer
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
        dashboard_port=8266,
        _temp_dir='/tmp/ray'  # Explicit temp directory
    )
    print(f"Ray initialized. Dashboard should be available at http://127.0.0.1:8265")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/Users/oussamayousr/Documents/ai-agent-multimodal-rag/data/colipali_data/page_1.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class VLMDeployment:
    def __init__(self):
        self.message = {'role': 'user', 'content': 'Why is the sky blue?'}

        self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        # Automatically select the best available device
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        torch.set_default_device(device)
        print(f"Using device: {device}")



        self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            device_map="auto" if device != "cpu" else None,
        )

        min_pixels = 224*224
        max_pixels = 1024*1024
        self.vl_model_processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)


            
       

    async def stream_tokens(self, messages,img_paths):
        # print("print the prompt and the image path : ", messages , img_paths)
        try:
            # Sampling parameters for response generation
            # chat_template = [
            #     {
            #         "role": "user",
            #         "content": [
            #             *[{ "type" : "image","image" : img_path} for img_path in img_paths ],
            #             {"type": "text", "text": messages},
      
                        
            #         ],
            #     }
            
            paths = [{ "type" : "image","image" : img_path} for img_path in img_paths ]
            chat_template = [
                {
                    "role": "user",
                    "content": paths + [{"type": "text", "text": messages}]
                              
                }]

            text = self.vl_model_processor.apply_chat_template(
                chat_template, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(chat_template)
            inputs = self.vl_model_processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Inference: Generation of the output

            generation_kwargs = dict(inputs, streamer=self.streamer, max_new_tokens=1500)

            thread = Thread(target=self.vl_model.generate, kwargs=generation_kwargs)
            thread.start()
            for output_text in self.streamer : 
                if output_text : 
                    yield output_text

        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    @app.post("/generate")
    async def generate(self, token: Annotated[str, Form()],files: Annotated[List[UploadFile], File()]):
      
        # prompt = payload.get("prompt")
        pictures_paths = []
        # TODO : Multi_process it 
        for file in files : 
            picture_path ='/Users/oussamayousr/Documents/ai-agent-multimodal-rag/data/vlm_uploaded_data/'+file.filename
            pictures_paths.append(picture_path)
            img = Image.open(file.file)
            try : 
                await run_in_threadpool(img.save, picture_path)
            except Exception as e : 
                print(f"Caught: {e}")
                raise ValueError("cannot save the image file!") 

        try:
            return StreamingResponse(
                self.stream_tokens(token,pictures_paths),
                media_type="text/event-stream"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
# Bind the deployment to an application
qwen_app = VLMDeployment.bind()

# 2: Deploy the application locally.
serve.run(qwen_app,route_prefix="/",
        name="vlm_deployment", blocking=True)