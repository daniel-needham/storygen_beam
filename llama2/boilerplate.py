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
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM, 
    AutoTokenizer
)

base_model = "mistralai/Mistral-7B-Instruct-v0.2"

app = App(
    name="mistral-7b-instruct",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        gpu="A10G",
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
                "vllm"
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
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        cache_dir="./model_weights",
        legacy=True,
        device_map={"": 0},
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        cache_dir="./model_weights",
        device_map={"": 0},
    )

    return model, tokenizer


@app.task_queue(outputs=[Output(path="output.txt")], loader=load_models)
def generate(**inputs):
    # Grab inputs passed to the API
    try:
        prompt = inputs["prompt"]
    # Use a default prompt if none is provided
    except KeyError:
        prompt = "The meaning of life is"


    model, tokenizer = load_models()

    tokenizer.bos_token_id = 1
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    generation_config = GenerationConfig(
        temperature=0.3,
        do_sample=True
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=600,
        )

    s = generation_output.sequences[0]
    decoded_output = tokenizer.decode(s, skip_special_tokens=True).strip()

    print(decoded_output)

    # Write text output to a text file, which we'll retrieve when the async task completes
    output_path = "output.txt"
    with open(output_path, "w") as f:
        f.write(decoded_output)
