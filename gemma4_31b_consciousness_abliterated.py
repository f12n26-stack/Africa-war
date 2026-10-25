from __future__ import annotations

from transformers import AutoModelForMultimodalLM, AutoProcessor, pipeline

MODEL_ID = "cosmicoptima/gemma4-31b-consciousness-abliterated"


def run_pipeline_example() -> object:
    """Run a minimal multimodal inference example with the high-level pipeline API."""
    pipe = pipeline("text-generation", model=MODEL_ID)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        }
    ]
    return pipe(text=messages)


def run_direct_model_example() -> str:
    """Run the same task with the lower-level AutoProcessor + AutoModelForMultimodalLM API."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, device_map="auto")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])


if __name__ == "__main__":
    print("=== pipeline example ===")
    print(run_pipeline_example())

    print("\n=== direct model example ===")
    print(run_direct_model_example())
