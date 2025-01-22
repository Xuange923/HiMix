from . import register_llm
# from transformers import LlamaForCausalLM, AutoTokenizer

from transformers import AutoTokenizer
from .modeling_files.modeling_llama import LlamaForCausalLM

@register_llm('llama-3')
def return_llama3class():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        return tokenizer
    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
