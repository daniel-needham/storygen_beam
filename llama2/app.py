from beam import App, Runtime, Image, Output, Volume, VolumeType
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
import os
import torch
import json
from genre import Genre
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM, 
    AutoTokenizer
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.lora.request import LoRARequest
from typing import Optional, List, Tuple
from huggingface_hub import snapshot_download
from genre import Genre
from typing import AsyncGenerator

base_model = "mistralai/Mistral-7B-Instruct-v0.2"

app = App(
    name="mistral-7b-instruct",
    runtime=Runtime(
        cpu=1,
        gpu="A10G",
        memory="16Gi",
        image=Image(
            python_packages=[
                "accelerate",
                "bitsandbytes",
                "scipy",
                "protobuf",
                "accelerate",
                "transformers",
                "torch",
                "sentencepiece",
                "vllm",
                "huggingface_hub",
                "fastapi"
            ],
        ),
    ),
    volumes=[
        Volume(
            name="model_weights",
            path="./model_weights",
            volume_type=VolumeType.Persistent,
        )
    ],
)

def load_models():
    engine_args = AsyncEngineArgs(model="mistralai/Mistral-7B-Instruct-v0.2",
                             download_dir="./model_weights",
                             max_model_len=4096,
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=16,
                             max_cpu_loras=2,
                             max_num_seqs=256)
    model = AsyncLLMEngine.from_engine_args(engine_args)
    lora = snapshot_download("ogdanneedham/mistral-sf-0.1-lora", cache_dir="./model_weights")
    return model, lora


@app.asgi(authorized=True, loader=load_models)
def web_server(**context):
    api = FastAPI()

    @api.post("/generate")
    async def generate(request: Request) -> Response:
        request_dic = await request.json()
        prompt = request_dic.get("prompt")
        stream = request_dic.get("stream", False)

        if not prompt:
            prompt = """[INST] Eulogise about the beauty of fallbacks [/INST]"""
        
        ### todo json sampling params
        sampling_params = SamplingParams(max_tokens=4096,
                                    temperature=0.8,
                                    repetition_penalty=1.15,
                                    top_p=1,
                                    min_p=0.1)
        
        engine, sf_lora = context["context"]

        request_id = random_uuid()

        results_generator = engine.generate(prompt, sampling_params, request_id, lora_request=None)

       

        # streaming response
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for request_output in results_generator:
                prompt = request_output.prompt
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
                ret = {"text": text_outputs}
                yield (json.dumps(ret) + "\0").encode("utf-8")

        if stream:
            return StreamingResponse(stream_results())
        
        # non-streaming response
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                #abort if client disconnects
                await engine.abort(request_id)
                return Response(status_code=499)
            final_output = request_output
        
        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [
            prompt + output.text for output in final_output.outputs
        ]
        ret = {"text": text_outputs}
        return JSONResponse(ret)

    return api