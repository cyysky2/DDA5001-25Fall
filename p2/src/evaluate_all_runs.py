import os
import sys
import json
import subprocess
from pathlib import Path
import csv

def run_rollout_and_eval(run_dir: Path,
                         base_model_name: str,
                         optimization_method: str,
                         rollout_script: str,
                         evaluate_script: str):
    """
    For a single run_dir:
        1) run rollout.py to generate MATH-500 outputs
        2) run evaluate.py to score them
    Returns the path to the scored JSONL file.
    """
    python_exe = sys.executable

    gen_path = run_dir / "math500_generations.jsonl"
    scored_path = run_dir / "math500_scored.jsonl"

    # --- 1) rollout.py ---
    if not gen_path.exists():
        print(f"[{run_dir.name}] Generations not found. Running rollout.py...")

        if optimization_method == "lora":
            # base model from finetune summary + LoRA adapter from run_dir
            cmd = [
                python_exe,
                rollout_script,
                "--model", base_model_name,
                "--output_file", str(gen_path),
                "--lora_path", str(run_dir),
            ]
        else:
            # full fine-tuned model saved in run_dir
            cmd = [
                python_exe,
                rollout_script,
                "--model", str(run_dir),
                "--output_file", str(gen_path),
            ]

        print("  rollout cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"[{run_dir.name}] Generations already exist, skipping rollout.")

    # --- 2) evaluate.py ---
    if not scored_path.exists():
        print(f"[{run_dir.name}] Scored file not found. Running evaluate.py...")
        cmd = [
            python_exe,
            evaluate_script,
            "--input_file", str(gen_path),
            "--output_file", str(scored_path),
        ]
        print("  evaluate cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"[{run_dir.name}] Scored file already exists, skipping evaluate.")

    return scored_path


def compute_accuracy_from_scored(scored_path: Path):
    """
    Reads math500_scored.jsonl and computes mean 'score'.
    Returns (accuracy, num_examples).
    """
    total_score = 0.0
    count = 0
    with scored_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # evaluate.py stores "score" in each line
            if "score" in data:
                total_score += float(data["score"])
                count += 1
    if count == 0:
        return None, 0
    return total_score / count, count


def main():
    base_output_dir = Path("out-instruction-tuning")   # where all run_* folders are
    rollout_script = "rollout.py"
    evaluate_script = "evaluate.py"

    if not base_output_dir.exists():
        print(f"Base output dir not found: {base_output_dir}")
        return

    # Collect per-run results
    all_results = []

    # Walk through subdirectories (each run)
    for run_dir in sorted(base_output_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"[{run_dir.name}] summary.json not found, skipping.")
            continue

        print("\n" + "="*80)
        print(f"Processing run: {run_dir.name}")
        print("="*80)

        # --- Load training summary / hyperparams ---
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        optimization_method = summary.get("optimization_method", "unknown")
        base_model_name = summary.get("model_name", "Qwen/Qwen3-0.6B-Base")

        # Run rollout + evaluate if needed
        scored_path = run_rollout_and_eval(
            run_dir=run_dir,
            base_model_name=base_model_name,
            optimization_method=optimization_method,
            rollout_script=rollout_script,
            evaluate_script=evaluate_script,
        )

        # Compute accuracy from scored JSONL
        accuracy, n = compute_accuracy_from_scored(scored_path)
        print(f"[{run_dir.name}] Accuracy: {accuracy:.4f} over {n} examples")

        # Merge into one record
        result = {
            "run_dir": run_dir.name,
            "optimization_method": optimization_method,
            "accuracy": accuracy,
            "num_examples": n,
            "best_val_loss": summary.get("best_val_loss"),
            "final_val_loss": summary.get("final_val_loss"),
            "train_time_sec": summary.get("train_time_sec"),
            "max_vram_gb": summary.get("max_vram_gb"),
            "learning_rate": summary.get("learning_rate"),
            "batch_size": summary.get("batch_size"),
            "grad_accumulation_steps": summary.get("grad_accumulation_steps"),
            "lora_rank": summary.get("lora_rank"),
            "lora_dropout": summary.get("lora_dropout"),
            "lora_target_modules": summary.get("lora_target_modules"),
            "modules_to_save": summary.get("modules_to_save"),
            "num_epochs": summary.get("num_epochs"),
        }
        all_results.append(result)

    # --- Save aggregate table to CSV ---
    if not all_results:
        print("No runs processed, nothing to save.")
        return

    agg_path = base_output_dir / "all_results.csv"
    fieldnames = list(all_results[0].keys())
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\nSaved aggregated results to: {agg_path}")

    # --- Print a quick summary by optimizer ---
    print("\n=== Accuracy by optimization_method ===")
    by_opt = {}
    for r in all_results:
        opt = r["optimization_method"]
        by_opt.setdefault(opt, []).append(r["accuracy"])

    for opt, accs in by_opt.items():
        mean_acc = sum(accs) / len(accs)
        print(f"{opt:>8}: mean accuracy = {mean_acc:.4f} over {len(accs)} runs")


if __name__ == "__main__":
    main()
