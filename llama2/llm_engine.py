from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput, LLM
from vllm.lora.request import LoRARequest
from typing import Optional, List, Tuple
from huggingface_hub import snapshot_download
from genre import Genre


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="mistralai/Mistral-7B-Instruct-v0.2",
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=16,
                             max_cpu_loras=2,
                             max_num_seqs=256)
    return LLMEngine.from_engine_args(engine_args)


class GenEngine:

    def __init__(self):
        self.engine = initialize_engine()
        self.sf_lora = snapshot_download("ogdanneedham/mistral-sf-0.1-lora")


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