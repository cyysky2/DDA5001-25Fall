import itertools
import subprocess
import sys
import os
from datetime import datetime

def main():

    python_exe = sys.executable
    ft_script = "finetune.py"   # Same folder as this script

    # ============================================================================
    # 1. ADJUSTABLE HYPERPARAMETERS (lists)
    #    To use default: keep the list as length 1 with the default value.
    #    To sweep: simply add more values into a list.
    # ============================================================================

    # ----- Optimizer setting -----
    optimization_methods = ["lora"]        # ["adam", "sgd", "lora"]

    # ----- Learning rate -----
    learning_rates = [0.0001]            # e.g., [2e-5, 5e-5]

    # ----- Batch size & grad accumulation -----
    batch_sizes = [4]                      # e.g., [2, 4]
    grad_accumulation_steps_list = [4]

    # ----- LoRA-specific hyperparameters -----
    lora_ranks = [8,16,32]                       # e.g., [4, 8, 16]
    lora_dropouts = [0.1, 0.2]              # e.g., [0.05, 0.1, 0.2]

    # LoRA target modules
    '''
    [
        "q_proj,v_proj",
        "q_proj,k_proj,v_proj,o_proj",
        "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
    ]
    '''
    lora_target_modules_list = [
        "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
    ]

    # Extra modules to save (full precision)
    modules_to_save_list = [
        "input_layernorm,post_attention_layernorm,norm"
    ]

    # ----- Training duration -----
    num_epochs_list = [2]                  # e.g., [1, 3]

    # ----- Model & data settings -----
    model_names = ["Qwen/Qwen3-0.6B-Base"]
    data_dirs = ["data"]
    output_dirs = ["out-instruction-tuning"]

    # ============================================================================
    # 2. Cartesian product of all hyperparameters
    # ============================================================================

    all_combinations = list(itertools.product(
        optimization_methods,
        learning_rates,
        batch_sizes,
        grad_accumulation_steps_list,
        lora_ranks,
        lora_dropouts,
        lora_target_modules_list,
        modules_to_save_list,
        num_epochs_list,
        model_names,
        data_dirs,
        output_dirs,
    ))

    print(f"Total experiment runs: {len(all_combinations)}")

    # ============================================================================
    # 3. Launch each experiment
    # ============================================================================

    for i, combo in enumerate(all_combinations, start=1):

        (
            optimization_method,
            lr,
            batch_size,
            grad_steps,
            lora_rank,
            lora_dropout,
            lora_target_modules,
            modules_to_save,
            num_epochs,
            model_name,
            data_dir,
            output_dir,
        ) = combo

        print("\n" + "="*90)
        print(f"Run {i}/{len(all_combinations)}")
        print(f"optimizer            = {optimization_method}")
        print(f"learning_rate        = {lr}")
        print(f"batch_size           = {batch_size}")
        print(f"grad_accum_steps     = {grad_steps}")
        print(f"lora_rank            = {lora_rank}")
        print(f"lora_dropout         = {lora_dropout}")
        print(f"lora_target_modules  = {lora_target_modules}")
        print(f"modules_to_save      = {modules_to_save}")
        print(f"num_epochs           = {num_epochs}")
        print(f"model_name           = {model_name}")
        print(f"data_dir             = {data_dir}")
        print(f"output_dir           = {output_dir}")
        print("="*90)

        cmd = [
            python_exe,
            ft_script,

            "--optimization_method", optimization_method,
            "--learning_rate", str(lr),
            "--batch_size", str(batch_size),
            "--grad_accumulation_steps", str(grad_steps),

            # LoRA settings
            "--lora_rank", str(lora_rank),
            "--lora_dropout", str(lora_dropout),
            "--lora_target_modules", lora_target_modules,
            "--modules_to_save", modules_to_save,

            "--model_name", model_name,
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--num_epochs", str(num_epochs),
        ]

        # Print full command for debugging
        print("Executing:")
        print(" ".join(cmd))

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[ERROR] Run {i} failed with code {result.returncode}, stopping.")
            break


if __name__ == "__main__":
    main()
