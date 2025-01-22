from . import register_llm
# from transformers import LlamaForCausalLM, AutoTokenizer

from transformers import AutoTokenizer
from .modeling_files.modeling_llama import LlamaForCausalLM

@register_llm('tinyllama')
def return_tinyllamaclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
