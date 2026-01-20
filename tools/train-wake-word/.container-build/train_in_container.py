import subprocess
import sys
import yaml
import shutil
import os
from pathlib import Path


def find_piper_sample_generator():
    """Find the piper-sample-generator installation path"""
    # Known location from Dockerfile
    known_path = Path("/opt/piper-sample-generator")
    if (known_path / "generate_samples.py").exists():
        print(f"Found piper-sample-generator at: {known_path}")
        return str(known_path)

    raise RuntimeError(f"piper-sample-generator not found at {known_path}")


def run_step(args, description):
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print("=" * 60)
    result = subprocess.run([sys.executable] + args, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with code {result.returncode}")


def main():
    config_path = Path("/app/config.yaml")
    output_dir = Path("/app/output")
    work_dir = Path("/app/work")
    output_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]

    # Find piper-sample-generator path
    piper_path = find_piper_sample_generator()
    print(f"Found piper-sample-generator at: {piper_path}")

    # Add required paths to config
    config["piper_sample_generator_path"] = piper_path
    config["output_dir"] = str(work_dir / "output")

    # Set defaults for missing config values
    config.setdefault("model_type", "dnn")
    config.setdefault("layer_size", 96)
    config.setdefault("tts_batch_size", 100)
    config.setdefault("augmentation_rounds", 3)
    config.setdefault("augmentation_batch_size", 100)
    config.setdefault("max_negative_weight", 1000)
    config.setdefault("target_false_positives_per_hour", 0.5)
    config.setdefault("batch_n_per_class", 500)

    # These paths need to exist - create empty dirs as placeholders if needed
    config.setdefault("rir_paths", [])
    config.setdefault("background_paths", [])
    config.setdefault("background_paths_duplication_rate", [])
    config.setdefault("false_positive_validation_data_path", "")
    config.setdefault("feature_data_files", {})

    # Write updated config
    updated_config_path = work_dir / "config_updated.yaml"
    with open(updated_config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Updated config written to: {updated_config_path}")
    print("Config contents:")
    print(yaml.dump(config, default_flow_style=False))

    # Find the openwakeword train.py script
    import openwakeword

    oww_dir = Path(openwakeword.__file__).parent
    train_script = oww_dir / "train.py"

    # Step 1: Generate synthetic clips
    run_step(
        [
            str(train_script),
            "--training_config",
            str(updated_config_path),
            "--generate_clips",
        ],
        "Step 1: Generating synthetic clips",
    )

    # Step 2: Augment clips
    run_step(
        [
            str(train_script),
            "--training_config",
            str(updated_config_path),
            "--augment_clips",
        ],
        "Step 2: Augmenting clips",
    )

    # Step 3: Train model
    run_step(
        [
            str(train_script),
            "--training_config",
            str(updated_config_path),
            "--train_model",
        ],
        "Step 3: Training model",
    )

    # Copy output model to mounted output directory
    model_output = work_dir / "output" / model_name / f"{model_name}.onnx"
    if model_output.exists():
        shutil.copy(model_output, output_dir / f"{model_name}.onnx")
        print(f"\nModel saved to /app/output/{model_name}.onnx")
    else:
        print(f"Warning: Expected model at {model_output} not found")
        # Try to find it
        for onnx in work_dir.rglob("*.onnx"):
            print(f"Found: {onnx}")
            shutil.copy(onnx, output_dir / onnx.name)


if __name__ == "__main__":
    main()
