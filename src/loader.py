import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

def load_model(model_id, precision):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if precision == "fp16":

        device = torch.device("mps")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(device)

    elif precision == "int8":

        device = torch.device("cpu")

        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            low_cpu_mem_usage=True
        )

    elif precision == "int4":

        device = torch.device("cpu")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            low_cpu_mem_usage=True
        )

    return model, tokenizer, device