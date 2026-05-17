import time
import pandas as pd

from metrics import get_ram_usage_gb
from utils import cleanup
from loader import load_model

from config import (
    PROMPT,
    MAX_NEW_TOKENS
)

def run_benchmark(model_id, precision):

    model, tokenizer, device = load_model(
        model_id,
        precision
    )

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    initial_ram = get_ram_usage_gb()

    start_time = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    end_time = time.time()

    final_ram = get_ram_usage_gb()

    generated_tokens = outputs.shape[1]

    tokens_per_second = generated_tokens / (
        end_time - start_time
    )

    results = {
        "model": model_id,
        "precision": precision,
        "inference_time": end_time - start_time,
        "tokens_per_second": tokens_per_second,
        "ram_usage_gb": final_ram - initial_ram
    }

    df = pd.DataFrame([results])
    
    safe_model_name = model_id.replace("/", "_")

    filename = f"../results/{safe_model_name}_{precision}"

    df.to_csv(filename, index=False)

    del model

    cleanup()

    return pd.DataFrame([results])