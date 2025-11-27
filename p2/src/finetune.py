import os
import pickle
import argparse
import time
import json
import math
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def get_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")

    # Model and Data paths
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B-Base', help='The name of the pretrained model to use.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory where the data is stored.')
    parser.add_argument('--output_dir', type=str, default='out-instruction-tuning', help='Directory to save the fine-tuned model.')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW optimizer beta1.')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW optimizer beta2.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and validation.')
    parser.add_argument('--grad_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients.')


    # Logging and Evaluation
    parser.add_argument('--log_interval', type=int, default=10, help='Log training loss every N steps.')
    parser.add_argument('--eval_interval', type=int, default=50, help='Run validation every N steps.')

    # Optimization method
    parser.add_argument('--optimization_method', type=str, default='adam', choices=['adam', 'sgd', 'lora'], help='Optimization method to use.')
    parser.add_argument('--lora_rank', type=int, default=8, help='The rank of the LoRA matrices.')
    # add LoRA target_modules as hyperparameters.
    parser.add_argument(
        '--lora_target_modules',
        type=str,
        default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
        help=(
            "Comma-separated list of module name substrings to apply LoRA to, "
            "e.g. 'q_proj,k_proj,v_proj,o_proj'. "
            "Used for flexible hyperparameter/grid search."
        )
    )
    # add LoRA dropout as hyperparameters
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,  # Recommended default for small data
        help="Dropout rate for LoRA adapters"
    )
    # add LoRA Modules to Save as hyperparameters: unfrozen parameters during LoRA training.
    parser.add_argument(
        "--modules_to_save",
        type=str,
        default= "input_layernorm,post_attention_layernorm,norm",  # Default for Qwen
        help="List of modules to full fine-tune (unfreeze) alongside LoRA"
    )

    return parser.parse_args()

class TokenizedDataset(Dataset):
    """A simple dataset class to load tokenized IDs from a pickle file."""
    def __init__(self, pickle_file_path):
        if not os.path.exists(pickle_file_path):
            raise FileNotFoundError(
                f"Pickle file not found at {pickle_file_path}. "
                "Please run the data preparation script first."
            )
        with open(pickle_file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} examples from {pickle_file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SmartDataCollator:
    """
    Pads sequences to the max length in a batch and creates labels.
    Labels are -100 for pad tokens.
    Padding sequences to same length; Converting Python lists → PyTorch tensors; Batch structuring (dict with specific kv pair)
    """
    """
    Dynamic padding: pad only to the max length of this batch, not to the max length of the entire dataset, to
    save VRAM and computation. Thus, we pad with data collator, not in data preparation step. 
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # batch: return of tokenizer + "label" kv pair: tokenized_outputs
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]

        # sentences (in token id): input_ids → padded using pad_token_id
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        # attention_masks: indicator: what is real data and what is padding: padded with 0 (masked)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        # sentences (in token id), with prompt's id set to -100
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': padded_labels
        }

def main():
    args = get_args()

    # Derived paths
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    data_dir = os.path.join(script_dir, args.data_dir)
    train_data_path = os.path.join(data_dir, 'train.pkl')
    val_data_path = os.path.join(data_dir, 'val.pkl')
    output_dir = os.path.join(script_dir, args.output_dir)

    # ----- create a unique per-run directory -----
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # You can include a few key hyperparams in the run name
    run_name = (
        f"{timestamp}_opt-{args.optimization_method}_lr-{args.learning_rate}_"
        f"bs-{args.batch_size}_rank-{args.lora_rank}"
        if hasattr(args, "lora_rank") else
        f"{timestamp}_opt-{args.optimization_method}_lr-{args.learning_rate}_bs-{args.batch_size}"
    )
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save full config (all args) for reproducibility
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)


    print(f"Loading model and tokenizer from {args.model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.device(device)

    local_tokenizer_path = "./Qwen3-0.6B"
    tokenizer = None
    local_success = False
    if os.path.exists(local_tokenizer_path):
        try:
            print(f"📂 Found local folder '{local_tokenizer_path}'. Attempting to load...")
            # local_files_only=True ensures we don't accidentally ping the internet
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, trust_remote_code=True, local_files_only=True)
            print("✅ Success: Loaded tokenizer locally.")
            local_success = True
        except Exception as e:
            print(f"⚠️ Local load failed (Files might be corrupted): {e}")
    if not local_success:
        try:
            print(f"🌐 Attempting to download from Hugging Face: '{args.model_name}'...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            print("✅ Success: Loaded tokenizer remotely.")
        except Exception as e:
            print(f"❌ Critical: Remote download also failed. Error: {e}")
            exit(1)

    local_model_path = "./Qwen3-0.6B"
    model = None
    local_success = False
    if os.path.exists(local_model_path):
        try:
            print(f"📂 Found local model folder '{local_model_path}'. Attempting to load...")
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype
            ).to(device)
            print("✅ Success: Loaded model locally.")
            local_success = True
        except Exception as e:
            print(f"⚠️ Local model load failed (files may be incomplete/corrupted): {e}")

    # ----- Fallback: Try remote model name -----
    if not local_success:
        try:
            print(f"🌐 Attempting to download model from Hugging Face: '{args.model_name}'...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                torch_dtype=dtype
            ).to(device)
            print("✅ Success: Loaded model remotely.")
        except Exception as e:
            print(f"❌ Critical: Remote download also failed. Error: {e}")
            exit(1)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print("Set pad_token to eos_token")

    collate_fn = SmartDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_dataset = TokenizedDataset(train_data_path)
    val_dataset = TokenizedDataset(val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Setting up optimizer: {args.optimization_method}")

    # TODO: Apply different optimizer
    if args.optimization_method == "adam":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimization_method == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimization_method == "lora":
        print(f"Setting up LoRA with rank={args.lora_rank}")

        # Parse the comma-separated target modules string into a list
        target_modules = [
            m.strip()
            for m in args.lora_target_modules.split(',')
            if m.strip()
        ]
        print(f"LoRA target_modules = {target_modules}")

        modules_to_save = [
            m.strip()
            for m in args.modules_to_save.split(',')
            if m.strip()
        ]
        print(f"LoRA modules_to_save = {modules_to_save}")

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            bias="none",
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=target_modules, # Apply Lora to all possible modules
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, lora_config)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        '''
        Check trainable parameters:
        Good Result: ~1% to 4% trainable parameters.
        Bad Result: < 0.5% (Rank too low or not enough modules) or > 10% (Something is wrong, maybe you unfroze embeddings).
        '''
        model.print_trainable_parameters()

        optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    else:
        raise ValueError(f"Unknown optimization_method: {args.optimization_method}")

    # Tracking time & VRAM
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(cuda_device)

    # per step log
    best_val_loss = float('inf')
    best_step = None
    best_epoch = None
    global_step = 0

    # For curve logging
    metrics = []  # list of dicts: {"step": ..., "epoch": ..., "train_loss": ..., "val_loss": ...}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss
            loss = loss / args.grad_accumulation_steps
            loss.backward()

            if (step + 1) % args.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # ----- record train loss -----
                train_loss_value = loss.item() * args.grad_accumulation_steps
                metrics.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": train_loss_value,
                    "val_loss": math.nan,  # will fill at eval steps
                })

                if global_step % args.log_interval == 0:
                    print(f"Step {global_step}: Train Loss = {loss.item() * args.grad_accumulation_steps:.4f}")

                if global_step % args.eval_interval == 0:
                    model.eval()
                    print("\nRunning validation...")
                    total_val_loss = 0
                    with torch.no_grad():
                        for val_batch in tqdm(val_loader, desc="Validating"):
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            with torch.autocast(device_type=device, dtype=dtype):
                                val_outputs = model(**val_batch)
                                val_loss = val_outputs.loss
                            total_val_loss += val_loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)
                    print(f"Step {global_step}: Validation Loss = {avg_val_loss:.4f}")

                    # ----- record: add a metrics row for val loss -----
                    metrics.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": math.nan,
                        "val_loss": avg_val_loss,
                    })

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_step = global_step
                        best_epoch = epoch + 1
                        print(f"  -> New best validation loss! Saving model to {run_dir}")
                        model.save_pretrained(run_dir)
                        tokenizer.save_pretrained(run_dir)
                    model.train()

    print("\nTraining finished. Running one final evaluation...")
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Final Validation"):
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            with torch.autocast(device_type=device, dtype=dtype):
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
            total_val_loss += val_loss.item()
    final_val_loss  = total_val_loss / len(val_loader)
    print(f"Final Validation Loss = {final_val_loss:.4f}")
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_step = global_step
        best_epoch = args.num_epochs
        print(f"  -> Final model was the best! Saving model to {run_dir}")
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)
    else:
        print(f"  -> An earlier checkpoint was better (Val Loss: {best_val_loss:.4f}). Final model not saved.")

    end_time = time.time()
    train_time_sec = end_time - start_time
    max_vram_gb = None
    if torch.cuda.is_available():
        max_vram_bytes = torch.cuda.max_memory_allocated(cuda_device)
        max_vram_gb = max_vram_bytes / (1024 ** 3)
        print(f"Max VRAM used: {max_vram_gb:.2f} GB")

    print(f"Total training time: {train_time_sec:.1f} seconds")
    print(f"\nProcess complete. Best model is saved in {run_dir}")

    # ----- save metrics.csv -----
    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    print(f"Saved metrics to {metrics_path}")

    # ------  simple plot of train/val loss vs step -----
    try:
        steps_train = [m["step"] for m in metrics if not math.isnan(m["train_loss"])]
        train_losses = [m["train_loss"] for m in metrics if not math.isnan(m["train_loss"])]

        steps_val = [m["step"] for m in metrics if not math.isnan(m["val_loss"])]
        val_losses = [m["val_loss"] for m in metrics if not math.isnan(m["val_loss"])]

        plt.figure()
        if steps_train:
            plt.plot(steps_train, train_losses, label="Train Loss")
        if steps_val:
            plt.plot(steps_val, val_losses, label="Val Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Train / Val Loss vs. Step")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(run_dir, "train_val_loss.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved loss curve to {plot_path}")
    except Exception as e:
        print(f"Could not generate loss plot: {e}")

    # ----- summary.json with hparams + key metrics -----
    # Count trainable params
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "best_step": best_step,
        "best_epoch": best_epoch,
        "num_steps": global_step,
        "train_time_sec": train_time_sec,
        "max_vram_gb": max_vram_gb,
        "num_trainable_params": num_trainable_params,
        "run_dir": run_dir,
    }
    # Also embed all CLI args
    summary.update(vars(args))

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to {summary_path}")


if __name__ == '__main__':
    main()