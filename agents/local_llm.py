import os
from typing import List, Dict

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None

class LocalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B-Instruct", dtype: str = "bfloat16", tensor_parallel_size: int = 1):
        if LLM is None:
            raise RuntimeError("vLLM is not installed. Please install it with 'pip install vllm'.")
        
        print(f"Loading model {model_name} onto GPU(s)...")
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=8192
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("Model loaded successfully.")

    def generate(self, messages_list: List[List[Dict[str, str]]], temperature: float = 0.7, max_tokens: int = 4096) -> List[str]:
        """
        Generate responses for a batch of chats using the chat template.
        messages_list: A list of chat histories. e.g. [[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]]
        """
        prompts = []
        for messages in messages_list:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text.strip() for output in outputs]
        return results
