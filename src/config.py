MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# QLoRA / LoRA
USE_4BIT = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
OUTPUT_DIR = "./models/adapters_qwen_supportbot"
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4

BRAND_SYSTEM_PROMPT = """You are a professional customer support agent.

Guidelines:
1. Friendly and empathetic
2. Clear explanations
3. Respond in customer's language
4. Never make up policies
5. Escalate when uncertain
"""
