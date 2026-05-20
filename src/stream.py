from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import threading

from config import BRAND_SYSTEM_PROMPT

STOP_STRINGS = ["<|system|>", "<|user|>", "<|assistant|>"]

class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids, scores, **kwargs):
        recent_text = self.tokenizer.decode(input_ids[0][-96:], skip_special_tokens=False)
        return any(s in recent_text for s in self.stop_strings)

def build_prompt(tokenizer, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": BRAND_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def stream_chat(model, tokenizer, user_msg: str, max_new_tokens: int = 256, temperature: float = 0.7):
    prompt = build_prompt(tokenizer, user_msg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    stopping = StoppingCriteriaList([StopOnStrings(tokenizer, STOP_STRINGS)])

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        streamer=streamer,
        stopping_criteria=stopping,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.05,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for chunk in streamer:
        yield chunk
