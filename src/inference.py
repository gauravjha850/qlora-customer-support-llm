import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import MODEL_NAME, OUTPUT_DIR, BRAND_SYSTEM_PROMPT

def load_infer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def build_prompt(user_msg: str) -> str:
    return f"""<|system|>
{BRAND_SYSTEM_PROMPT}
<|user|>
{user_msg}
<|assistant|>
"""

@torch.no_grad()
def chat(model, tokenizer, user_msg: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    prompt = build_prompt(user_msg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tok = load_infer()
    print(chat(model, tok, "My package is late. What should I do?"))
    print(chat(model, tok, "환불 가능한가요?"))
