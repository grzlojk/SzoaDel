import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
from itertools import islice
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("WARNING: Hugging Face token not found. You may face rate-limiting issues.")

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
LAYER_IDX = 14  # (28 total)
NUM_TOKENS_WANTED = 200_000
BATCH_SIZE = 1  # batch has to be 1, because qwen is tweaking
SEQ_LEN = 256
SAVE_PATH = "data/activations/activations_qwen_fineweb.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model and tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer.pad_token to tokenizer.eos_token")

    print(f"Streaming dataset {DATASET_ID}...")
    dataset = load_dataset(
        DATASET_ID, name="sample-10BT", split="train", streaming=True
    )

    activations = []

    def hook_fn(module, inputs, outputs):
        hidden_states = outputs[0]
        activations.append(hidden_states.detach().cpu().to(torch.float32))

    try:
        layer = model.model.layers[LAYER_IDX]
        handle = layer.register_forward_hook(hook_fn)
    except Exception as e:
        print(f"ERROR: Could not register hook on 'model.model.layers[{LAYER_IDX}]'.")
        print("Please inspect the model architecture to find the correct path.")
        raise e

    tokens_collected = 0
    collected_token_ids = []

    # Use itertools.islice for a clean exit from the streaming dataset
    num_samples_to_process = (NUM_TOKENS_WANTED // (BATCH_SIZE * SEQ_LEN)) + 1
    sliced_dataset = islice(dataset, num_samples_to_process)

    print("Starting activation extraction...")
    with torch.no_grad():
        for sample in tqdm(
            sliced_dataset, total=num_samples_to_process, desc="Processing samples"
        ):
            text = sample.get("text")
            if not isinstance(text, str) or len(text) < 100:
                continue

            # Tokenize a single text sample (since BATCH_SIZE = 1)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=SEQ_LEN,
                padding="max_length",
            ).to(device)

            model(**inputs)
            collected_token_ids.append(inputs.input_ids.cpu())

    handle.remove()

    print("\n--- Post-processing ---")

    all_acts = torch.cat(activations, dim=0)
    all_ids = torch.cat(collected_token_ids, dim=0)

    # If the hook returned activations without a batch dim, reshape them
    if len(all_acts.shape) == 2 and len(all_ids.shape) == 2:
        expected_shape = (all_ids.shape[0], all_ids.shape[1], all_acts.shape[-1])
        all_acts = all_acts.reshape(expected_shape)

    # Flatten both tensors to get a list of activations and corresponding token IDs
    all_acts_flat = all_acts.view(-1, all_acts.shape[-1])
    all_ids_flat = all_ids.view(-1)

    assert all_acts_flat.shape[0] == all_ids_flat.shape[0], (
        "Mismatch after final flattening!"
    )

    # Filter out padding tokens
    mask = all_ids_flat != tokenizer.pad_token_id
    clean_acts = all_acts_flat[mask]
    clean_ids = all_ids_flat[mask]

    print(f"Total tokens collected (after padding removed): {clean_acts.shape[0]}")
    print(f"Final activations tensor shape: {clean_acts.shape}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "activations": clean_acts,
            "token_ids": clean_ids,
            "model_id": MODEL_ID,
            "layer_idx": LAYER_IDX,
        },
        SAVE_PATH,
    )
    print(f"\nSuccessfully saved data to {SAVE_PATH}")


if __name__ == "__main__":
    main()
