import os
import shutil
import zipfile
import json
import csv
from pathlib import Path

# ================================
# User settings
# ================================
RUNS_DIR = Path("out-instruction-tuning")
SAVE_DIR = Path("save_result")
SAVE_DIR.mkdir(exist_ok=True)

# Files we WANT to collect (experiment-generated)
KEEP_SUFFIXES = [
    ".json", ".jsonl", ".csv", ".png"
]

# Exact filenames to include (even if lacking suffix)
KEEP_EXACT = {
    "summary.json",
    "config.json",
    "metrics.csv",
    "train_val_loss.png",
}

# Patterns indicating checkpoint/model/internal files → skip
SKIP_KEYWORDS = [
    "pytorch_model",
    "model.safetensors",
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer",
    "vocab",
    "merges",
    "special_tokens",
    "added_tokens",
    "chat_template",
    "README",
]

# For "missing files" verification (per run)
EXPECTED_FILES_BASIC = [
    "summary.json",
    "config.json",
    "metrics.csv",
    "train_val_loss.png",
]
# For evaluation: expect at least one scored file
EXPECTED_SCORED_GLOB = "math500_scored*.jsonl"


def should_keep_file(filepath: Path) -> bool:
    """Return True if this is an experiment-generated file worth saving."""
    name = filepath.name

    # Skip checkpoint-like files, tokenizer files, model weights
    for kw in SKIP_KEYWORDS:
        if kw in name:
            return False

    # Keep specific known files
    if name in KEEP_EXACT:
        return True

    # Keep results based on extension
    if filepath.suffix.lower() in KEEP_SUFFIXES:
        return True

    return False


def extract_metadata(run_dir: Path) -> dict | None:
    """
    Extract useful metadata from summary.json for this run.
    Returns a dict (one CSV row) or None if summary.json is missing.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    md = {
        "run_dir": run_dir.name,
        "optimization_method": summary.get("optimization_method"),
        "learning_rate": summary.get("learning_rate"),
        "batch_size": summary.get("batch_size"),
        "grad_accumulation_steps": summary.get("grad_accumulation_steps"),
        "lora_rank": summary.get("lora_rank"),
        "lora_dropout": summary.get("lora_dropout"),
        "lora_target_modules": summary.get("lora_target_modules"),
        "modules_to_save": summary.get("modules_to_save"),
        "num_epochs": summary.get("num_epochs"),
        "best_val_loss": summary.get("best_val_loss"),
        "final_val_loss": summary.get("final_val_loss"),
        "best_step": summary.get("best_step"),
        "best_epoch": summary.get("best_epoch"),
        "num_steps": summary.get("num_steps"),
        "train_time_sec": summary.get("train_time_sec"),
        "max_vram_gb": summary.get("max_vram_gb"),
        "num_trainable_params": summary.get("num_trainable_params"),
    }
    return md


def verify_missing(run_dir: Path) -> list[tuple[str, str]]:
    """
    Check for missing key files in this run.
    Returns a list of (run_name, missing_description).
    """
    missing = []
    run_name = run_dir.name

    # Basic expected files
    for fname in EXPECTED_FILES_BASIC:
        if not (run_dir / fname).exists():
            missing.append((run_name, f"Missing {fname}"))

    # Evaluation expected: at least one scored JSONL
    scored_files = list(run_dir.glob(EXPECTED_SCORED_GLOB))
    if len(scored_files) == 0:
        missing.append((run_name, f"No '{EXPECTED_SCORED_GLOB}' files found"))

    return missing


def copy_run_results(run_dir: Path) -> dict | None:
    """
    Copy useful result files from one run folder into save_result/<run_name>/.
    Also returns metadata dict extracted from summary.json (or None).
    """
    run_name = run_dir.name
    dest_dir = SAVE_DIR / run_name
    dest_dir.mkdir(exist_ok=True)

    print(f"\nProcessing run: {run_name}")

    for item in run_dir.iterdir():
        if item.is_file():
            if should_keep_file(item):
                print(f"  + Copy: {item.name}")
                shutil.copy(item, dest_dir)
            else:
                print(f"  - Skip: {item.name}")
        # ignore subdirectories

    print(f"Finished collecting for run: {run_name}")
    return extract_metadata(run_dir)


def write_experiment_overview(metadata_rows: list[dict]):
    """
    Write all metadata rows into save_result/experiment_overview.csv
    """
    if not metadata_rows:
        print("\nNo metadata to write (no summary.json files found).")
        return

    overview_path = SAVE_DIR / "experiment_overview.csv"
    # Collect all keys across rows so we don't miss anything
    fieldnames = set()
    for row in metadata_rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)

    with overview_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(row)

    print(f"\nWrote experiment overview CSV to: {overview_path}")


def write_missing_report(missing_entries: list[tuple[str, str]]):
    """
    Write missing-files info to save_result/missing_files_report.txt
    """
    if not missing_entries:
        print("\nNo missing files detected.")
        return

    report_path = SAVE_DIR / "missing_files_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        for run_name, msg in missing_entries:
            line = f"{run_name}: {msg}"
            print(line)
            f.write(line + "\n")

    print(f"\nMissing files report written to: {report_path}")


def zip_results(output_zip="all_experiment_results.zip"):
    """
    Zip the entire save_result/ directory and also include out-instruction-tuning/all_results.csv if present.
    """
    zip_path = Path(output_zip)
    print(f"\nZipping results into: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add everything under save_result/
        for root, _, files in os.walk(SAVE_DIR):
            for fn in files:
                full_path = Path(root) / fn
                rel_path = full_path.relative_to(SAVE_DIR)
                zf.write(full_path, arcname=rel_path)

        # Also add all_results.csv if it exists in RUNS_DIR
        all_results_path = RUNS_DIR / "all_results.csv"
        if all_results_path.exists():
            zf.write(all_results_path, arcname="all_results.csv")
            print(f"Included all_results.csv from {all_results_path}")
        else:
            print("No all_results.csv found in runs dir; skipping.")

    print(f"Created ZIP: {zip_path}")


def main():
    if not RUNS_DIR.exists():
        print(f"Run directory {RUNS_DIR} not found.")
        return

    metadata_rows: list[dict] = []
    missing_entries: list[tuple[str, str]] = []

    # Go through each run folder
    for run_folder in sorted(RUNS_DIR.iterdir()):
        if run_folder.is_dir():
            # Copy result files
            md = copy_run_results(run_folder)
            if md is not None:
                metadata_rows.append(md)

            # Verify missing files
            missing_entries.extend(verify_missing(run_folder))

    # Write overview CSV
    write_experiment_overview(metadata_rows)

    # Write missing-files report
    write_missing_report(missing_entries)

    # Zip everything
    zip_results()


if __name__ == "__main__":
    main()
