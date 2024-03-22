"""
### Llama 2 ###

** Pre-requisites **


** Deploy the API **

```sh
beam deploy app.py:generate
```
"""
from beam import App, Runtime, Image, Output, Volume, VolumeType

import os
import torch
from genre import Genre
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM, 
    AutoTokenizer
)
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput, LLM
from vllm.lora.request import LoRARequest
from typing import Optional, List, Tuple
from huggingface_hub import snapshot_download
from genre import Genre

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
                "huggingface_hub"
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
    engine_args = EngineArgs(model="mistralai/Mistral-7B-Instruct-v0.2",
                             download_dir="./model_weights",
                             max_model_len=4096,
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=16,
                             max_cpu_loras=2,
                             max_num_seqs=256)
    model = LLMEngine.from_engine_args(engine_args)
    lora = snapshot_download("ogdanneedham/mistral-sf-0.1-lora", cache_dir="./model_weights")
    return model, lora

class GenEngine:

    def __init__(self, engine, sf_lora):
        self.engine = engine
        self.sf_lora = sf_lora


    ### todo add the other genres
    def convert_genre_to_lora(self, test_prompts: List[Tuple[str, SamplingParams,
                         Optional[Genre]]]) -> List[Tuple[str, SamplingParams,
                         Optional[LoRARequest]]]:
        """Convert genre to LoRARequest."""
        lora_requests = []
        for prompt, sampling_params, genre in test_prompts:
            if genre == Genre.SCIENCE_FICTION:
                lora_requests.append((prompt, sampling_params, LoRARequest("sf-lora", 1, self.sf_lora)))
            else:
                lora_requests.append((prompt, sampling_params, None))
        return lora_requests
    def process_requests(self,
                         test_prompts: List[Tuple[str, SamplingParams,
                         Optional[Genre]]]):
        """Continuously process a list of prompts and handle the outputs."""
        request_outputs = []
        request_id = 0
        test_prompts = self.convert_genre_to_lora(test_prompts)

        while test_prompts or self.engine.has_unfinished_requests():
            if test_prompts:
                prompt, sampling_params, lora_request = test_prompts.pop(0)
                self.engine.add_request(str(request_id),
                                   prompt,
                                   sampling_params,
                                   lora_request=lora_request)
                request_id += 1

            request_outputs: List[RequestOutput] = self.engine.step()

        return request_outputs


@app.task_queue(outputs=[Output(path="output.txt")], loader=load_models)
def generate(**inputs):
    # Grab inputs passed to the API
    try:
        prompt = inputs["prompt"]
    # Use a default prompt if none is provided
    except KeyError:
        prompt = """[INST] You are a renowned creative writer specialising in the genre of science fiction. Write a narrative section in the third person, where a man discovers a locked book, which may contain answers to the mystery of the stately home he is currently staying in. There is a strange chill in the air, as the suspicious maid enters the parlour[/INST]"""

    engine, sf_lora = inputs["context"]
    # Load the model
    gen = GenEngine(engine, sf_lora)


    sampling_params = SamplingParams(max_tokens=4096,
                                    temperature=0.8,
                                    repetition_penalty=1.15,
                                    top_p=1,
                                    min_p=0.1)

    # Generate the story
    output = gen.process_requests([(prompt, sampling_params, Genre.SCIENCE_FICTION)])
    output = output[0].outputs[0].text

    # Write text output to a text file, which we'll retrieve when the async task completes
    output_path = "output.txt"
    with open(output_path, "w") as f:
        f.write(output)

    return {"output": output}
