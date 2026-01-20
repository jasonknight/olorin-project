#!/usr/bin/env python3
"""
OpenWakeWord Custom Wake Word Training Tool

This script automates the entire process of training a custom wake word model:
1. Sets up Python 3.10 environment via pyenv
2. Installs OpenWakeWord training dependencies
3. Generates synthetic training data
4. Trains the base wake word model
5. Records user voice samples for custom verifier
6. Trains the verifier model for improved accuracy

Usage:
    ./train.py

Requirements:
    - pyenv installed and in PATH
    - Linux (for Piper TTS) or Docker available
    - Microphone for verifier recording
"""

import argparse
import logging
import os
import sys
import subprocess
import shutil
import json
from datetime import datetime
from pathlib import Path

# Constants
PYTHON_VERSION = "3.10.13"
VENV_NAME = ".venv-openwakeword"
SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / "models"
RECORDINGS_DIR = SCRIPT_DIR / "recordings"
CONFIG_DIR = SCRIPT_DIR / "configs"
LOG_DIR = SCRIPT_DIR / "logs"

# Global logger
logger: logging.Logger = None


def setup_logging(debug: bool = False) -> logging.Logger:
    """Set up logging with file and console handlers."""
    global logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create logger
    logger = logging.getLogger("train-wake-word")
    logger.setLevel(logging.DEBUG)  # Capture all levels
    logger.handlers.clear()

    # File handler - always DEBUG level for full diagnostics
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - INFO or DEBUG based on flag
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_formatter = logging.Formatter("%(levelname)-8s | %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.debug(f"Debug mode: {debug}")
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Platform: {sys.platform}")
    logger.debug(f"Script directory: {SCRIPT_DIR}")
    logger.debug(f"Working directory: {os.getcwd()}")

    return logger


def run_cmd(
    cmd: list[str] | str,
    check: bool = True,
    capture: bool = False,
    env: dict = None,
    description: str = None,
) -> subprocess.CompletedProcess:
    """Run a command with proper error handling and logging."""
    shell = isinstance(cmd, str)
    merged_env = {**os.environ, **(env or {})}

    cmd_str = cmd if shell else " ".join(cmd)
    desc = description or cmd_str[:80]

    logger.debug(f"Running command: {cmd_str}")
    if env:
        logger.debug(f"Additional env vars: {list(env.keys())}")

    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            capture_output=capture,
            text=True,
            env=merged_env,
        )

        if capture:
            if result.stdout:
                logger.debug(f"stdout: {result.stdout[:500]}...")
            if result.stderr:
                logger.debug(f"stderr: {result.stderr[:500]}...")

        logger.debug(f"Command completed with return code: {result.returncode}")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {desc}")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Command not found: {cmd_str}")
        logger.error(f"Error: {e}")
        raise


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_step(step: int, text: str):
    """Print a step indicator."""
    logger.info(f"[Step {step}] {text}")
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


def check_pyenv() -> bool:
    """Check if pyenv is installed and available."""
    pyenv_path = shutil.which("pyenv")
    logger.debug(f"Checking for pyenv: {pyenv_path}")
    if pyenv_path:
        logger.debug(f"pyenv found at: {pyenv_path}")
        # Log pyenv version
        try:
            result = run_cmd(["pyenv", "--version"], capture=True, check=False)
            logger.debug(f"pyenv version: {result.stdout.strip()}")
        except Exception as e:
            logger.debug(f"Could not get pyenv version: {e}")
    else:
        logger.warning("pyenv not found in PATH")
    return pyenv_path is not None


def check_python_version_installed(version: str) -> bool:
    """Check if a specific Python version is installed via pyenv."""
    logger.debug(f"Checking if Python {version} is installed via pyenv")
    result = run_cmd(["pyenv", "versions", "--bare"], capture=True, check=False)
    installed_versions = result.stdout.strip().split("\n")
    logger.debug(f"Installed Python versions: {installed_versions}")
    is_installed = version in result.stdout
    logger.debug(f"Python {version} installed: {is_installed}")
    return is_installed


def install_python_version(version: str):
    """Install Python version via pyenv."""
    logger.info(f"Installing Python {version} via pyenv...")
    print(f"Installing Python {version} via pyenv...")
    run_cmd(["pyenv", "install", "-s", version], description=f"pyenv install {version}")
    logger.info(f"Python {version} installation complete")


def get_python_path(version: str) -> str:
    """Get the path to a pyenv-installed Python."""
    logger.debug(f"Getting Python path for version {version}")
    result = run_cmd(
        f"pyenv prefix {version}", capture=True, description="Get pyenv prefix"
    )
    prefix = result.stdout.strip()
    python_path = f"{prefix}/bin/python"
    logger.debug(f"Python path: {python_path}")

    # Verify the path exists
    if not Path(python_path).exists():
        logger.error(f"Python executable not found at: {python_path}")
        raise FileNotFoundError(f"Python not found at {python_path}")

    return python_path


def create_venv(python_path: str, venv_path: Path):
    """Create a virtual environment."""
    logger.info(f"Creating virtual environment at {venv_path}")
    logger.debug(f"Using Python: {python_path}")
    print(f"Creating virtual environment at {venv_path}...")
    run_cmd(
        [python_path, "-m", "venv", str(venv_path)], description="Create virtualenv"
    )
    logger.info("Virtual environment created successfully")

    # Verify venv was created
    venv_python = venv_path / "bin" / "python"
    if not venv_python.exists():
        logger.error(f"Venv Python not found at: {venv_python}")
        raise RuntimeError(f"Failed to create venv - Python not found at {venv_python}")
    logger.debug(f"Venv Python verified at: {venv_python}")


def get_venv_python(venv_path: Path) -> str:
    """Get the Python executable in a venv."""
    python_path = str(venv_path / "bin" / "python")
    logger.debug(f"Venv Python path: {python_path}")
    return python_path


def install_dependencies(venv_python: str):
    """Install OpenWakeWord training dependencies."""
    logger.info("Starting dependency installation")
    print("Installing dependencies (this may take several minutes)...")

    # Core dependencies
    deps = [
        "torch",
        "torchaudio",
        "torchinfo",
        "torchmetrics",
        "tensorflow",
        "speechbrain",
        "audiomentations",
        "datasets",
        "mutagen",
        "pyyaml",
        "tqdm",
        "numpy",
        "scipy",
        "sounddevice",
        "soundfile",
    ]

    logger.debug(f"Dependencies to install: {deps}")

    logger.info("Upgrading pip...")
    run_cmd(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
        description="Upgrade pip",
    )

    logger.info(f"Installing {len(deps)} core dependencies...")
    run_cmd(
        [venv_python, "-m", "pip", "install"] + deps,
        description="Install core dependencies",
    )

    # Install OpenWakeWord with training support
    logger.info("Installing OpenWakeWord...")
    print("Installing OpenWakeWord...")
    run_cmd(
        [venv_python, "-m", "pip", "install", "openwakeword"],
        description="Install openwakeword",
    )

    # Try to install piper-sample-generator (Linux only)
    logger.info("Attempting to install piper-sample-generator (Linux only)...")
    print("Attempting to install piper-sample-generator...")
    result = run_cmd(
        [venv_python, "-m", "pip", "install", "piper-sample-generator"],
        check=False,
        capture=True,
        description="Install piper-sample-generator",
    )
    if result.returncode != 0:
        logger.warning("piper-sample-generator failed to install (requires Linux)")
        logger.debug(f"piper install stderr: {result.stderr}")
        print("Warning: piper-sample-generator failed to install (requires Linux)")
        print("Will use alternative training method.")
        return False

    logger.info("All dependencies installed successfully (including piper)")
    return True


def prompt_wake_word() -> tuple[str, str]:
    """Prompt user for wake word configuration."""
    print_header("Wake Word Configuration")

    print("Enter your desired wake word or phrase.")
    print("Tips:")
    print("  - 2-3 syllables work best (e.g., 'hey olorin', 'ok computer')")
    print("  - Avoid common words that appear in normal speech")
    print("  - Use distinct sounds that are easy to recognize")
    print()

    wake_word = input("Wake word/phrase: ").strip().lower()
    if not wake_word:
        wake_word = "hey olorin"
        print(f"Using default: {wake_word}")

    # Generate model name from wake word
    model_name = wake_word.replace(" ", "_").replace("'", "")

    logger.info(f"Wake word configured: '{wake_word}' -> model: {model_name}")

    print(f"\nWake word: '{wake_word}'")
    print(f"Model name: {model_name}")

    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm and confirm != "y":
        logger.debug("User chose to re-enter wake word")
        return prompt_wake_word()

    return wake_word, model_name


def generate_training_config(
    wake_word: str, model_name: str, config_path: Path
) -> dict:
    """Generate YAML training configuration."""
    logger.info(f"Generating training config for '{wake_word}'")

    config = {
        "target_phrase": [wake_word],
        "model_name": model_name,
        "n_samples": 2000,
        "n_samples_val": 500,
        "steps": 10000,
        "target_accuracy": 0.7,
        "target_recall": 0.5,
    }

    logger.debug(f"Training config: {config}")

    import yaml

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Training config saved to: {config_path}")
    print(f"Training config saved to: {config_path}")
    return config


def get_container_runtime() -> str | None:
    """Get available container runtime (podman preferred, then docker)."""
    logger.debug("Checking for container runtime...")

    podman_path = shutil.which("podman")
    if podman_path:
        logger.debug(f"Found podman at: {podman_path}")
        # Log podman version
        try:
            result = run_cmd(["podman", "--version"], capture=True, check=False)
            logger.debug(f"podman version: {result.stdout.strip()}")
        except Exception as e:
            logger.debug(f"Could not get podman version: {e}")
        return "podman"

    docker_path = shutil.which("docker")
    if docker_path:
        logger.debug(f"Found docker at: {docker_path}")
        try:
            result = run_cmd(["docker", "--version"], capture=True, check=False)
            logger.debug(f"docker version: {result.stdout.strip()}")
        except Exception as e:
            logger.debug(f"Could not get docker version: {e}")
        return "docker"

    logger.warning("No container runtime found (neither podman nor docker)")
    return None


def check_platform_support() -> str:
    """Check platform and return training method."""
    import platform

    system = platform.system().lower()
    machine = platform.machine()
    release = platform.release()

    logger.debug(f"Platform: system={system}, machine={machine}, release={release}")

    if system == "linux":
        logger.info("Linux detected - native training supported")
        return "native"
    elif system == "darwin":
        logger.info("macOS detected - checking for container runtime")
        runtime = get_container_runtime()
        if runtime:
            logger.info(f"Container runtime available: {runtime}")
            return "container"
        else:
            logger.info("No container runtime - will use Google Colab instructions")
            return "colab"
    else:
        logger.info(
            f"Unsupported platform: {system} - will use Google Colab instructions"
        )
        return "colab"


def train_with_container(wake_word: str, model_name: str, venv_python: str):
    """Train using container (Podman or Docker) for Linux compatibility."""
    logger.info("Starting container-based training")

    runtime = get_container_runtime()
    if not runtime:
        logger.error("No container runtime found")
        print("Error: No container runtime found (podman or docker)")
        return

    logger.info(f"Using container runtime: {runtime}")
    print_header(f"Training with {runtime.capitalize()} (Linux container)")

    dockerfile_content = """
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    libsndfile1 \\
    build-essential \\
    cmake \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch 2.x from official PyTorch index (required by piper-sample-generator)
RUN pip install --no-cache-dir \\
    torch==2.1.0 \\
    torchaudio==2.1.0 \\
    --index-url https://download.pytorch.org/whl/cpu

# Create constraints file to prevent torch/torchaudio from being changed
# Also pin numpy<2 for compatibility with pre-compiled packages
RUN echo "torch==2.1.0" > /tmp/constraints.txt && \\
    echo "torchaudio==2.1.0" >> /tmp/constraints.txt && \\
    echo "numpy<2" >> /tmp/constraints.txt

# Install numpy 1.x first
RUN pip install --no-cache-dir "numpy<2"

# Install openwakeword from source (train module not included in PyPI package)
RUN pip install --no-cache-dir -c /tmp/constraints.txt \\
    git+https://github.com/dscripka/openWakeWord.git

# Install remaining training dependencies
# Using speechbrain>=1.0 which is compatible with torchaudio 2.x
RUN pip install --no-cache-dir -c /tmp/constraints.txt \\
    "speechbrain>=1.0" \\
    torchinfo \\
    torchmetrics \\
    tensorflow \\
    audiomentations \\
    torch-audiomentations \\
    datasets \\
    mutagen \\
    pronouncing \\
    deep-phonemizer \\
    acoustics \\
    pyyaml \\
    onnx \\
    tf2onnx

# piper-sample-generator v2.0.0 (has default model, compatible with openwakeword)
# Clone it so we have the generate_samples.py script available
RUN git clone --branch v2.0.0 --depth 1 https://github.com/rhasspy/piper-sample-generator.git /opt/piper-sample-generator && \\
    cd /opt/piper-sample-generator && \\
    pip install --no-cache-dir -c /tmp/constraints.txt -e . && \\
    ls -la /opt/piper-sample-generator/

COPY train_in_container.py /app/
COPY config.yaml /app/

CMD ["python", "train_in_container.py"]
"""

    container_train_script = """
import subprocess
import sys
import yaml
import shutil
import os
from pathlib import Path

def find_piper_sample_generator():
    '''Find the piper-sample-generator installation path'''
    # Known location from Dockerfile
    known_path = Path("/opt/piper-sample-generator")
    if (known_path / "generate_samples.py").exists():
        print(f"Found piper-sample-generator at: {known_path}")
        return str(known_path)

    raise RuntimeError(f"piper-sample-generator not found at {known_path}")

def run_step(args, description):
    print(f"\\n{'='*60}")
    print(f"{description}")
    print('='*60)
    result = subprocess.run(
        [sys.executable] + args,
        capture_output=False
    )
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
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Updated config written to: {updated_config_path}")
    print(f"Config contents:")
    print(yaml.dump(config, default_flow_style=False))

    # Find the openwakeword train.py script
    import openwakeword
    oww_dir = Path(openwakeword.__file__).parent
    train_script = oww_dir / "train.py"

    # Step 1: Generate synthetic clips
    run_step([
        str(train_script),
        "--training_config", str(updated_config_path),
        "--generate_clips"
    ], "Step 1: Generating synthetic clips")

    # Step 2: Augment clips
    run_step([
        str(train_script),
        "--training_config", str(updated_config_path),
        "--augment_clips"
    ], "Step 2: Augmenting clips")

    # Step 3: Train model
    run_step([
        str(train_script),
        "--training_config", str(updated_config_path),
        "--train_model"
    ], "Step 3: Training model")

    # Copy output model to mounted output directory
    model_output = work_dir / "output" / model_name / f"{model_name}.onnx"
    if model_output.exists():
        shutil.copy(model_output, output_dir / f"{model_name}.onnx")
        print(f"\\nModel saved to /app/output/{model_name}.onnx")
    else:
        print(f"Warning: Expected model at {model_output} not found")
        # Try to find it
        for onnx in work_dir.rglob("*.onnx"):
            print(f"Found: {onnx}")
            shutil.copy(onnx, output_dir / onnx.name)

if __name__ == "__main__":
    main()
"""

    # Create temporary directory for container build
    build_dir = SCRIPT_DIR / ".container-build"
    logger.debug(f"Creating build directory: {build_dir}")
    build_dir.mkdir(exist_ok=True)

    # Write files
    dockerfile_path = build_dir / "Dockerfile"
    script_path = build_dir / "train_in_container.py"
    logger.debug(f"Writing Dockerfile to: {dockerfile_path}")
    dockerfile_path.write_text(dockerfile_content)
    logger.debug(f"Writing training script to: {script_path}")
    script_path.write_text(container_train_script)

    # Copy config
    config_path = CONFIG_DIR / f"{model_name}.yaml"
    dest_config = build_dir / "config.yaml"
    logger.debug(f"Copying config from {config_path} to {dest_config}")
    shutil.copy(config_path, dest_config)

    # Build container
    logger.info(f"Building container image with {runtime}...")
    print(f"Building container image with {runtime}...")
    run_cmd(
        [runtime, "build", "-t", "openwakeword-trainer", str(build_dir)],
        description=f"{runtime} build",
    )

    # Run training
    logger.info("Running training in container...")
    print("Running training in container...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {MODELS_DIR}")

    run_cmd(
        [
            runtime,
            "run",
            "--rm",
            "-v",
            f"{MODELS_DIR}:/app/output",
            "openwakeword-trainer",
        ],
        description=f"{runtime} run training",
    )

    # Cleanup
    logger.debug(f"Cleaning up build directory: {build_dir}")
    shutil.rmtree(build_dir)

    model_path = MODELS_DIR / f"{model_name}.onnx"
    logger.info(f"Container training complete. Model saved to: {model_path}")
    print(f"\nModel saved to: {model_path}")


def print_colab_instructions(wake_word: str, model_name: str):
    """Print instructions for training via Google Colab."""
    logger.info("Displaying Google Colab training instructions")
    print_header("Google Colab Training Instructions")

    print("""
Since you're on macOS and Piper TTS requires Linux, you'll need to use
Google Colab for the synthetic data generation and training.

Steps:
1. Open the OpenWakeWord training notebook:
   https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb

2. In the notebook, modify the configuration cell to use your wake word:
""")

    print(f'''   target_phrase: ["{wake_word}"]
   model_name: "{model_name}"
   n_samples: 2000
   n_samples_val: 500
   steps: 10000
''')

    print("""
3. Run all cells in the notebook (takes ~45 minutes)

4. Download the generated model files:
   - {model_name}.onnx (or .tflite)

5. Place the model in:
""")
    print(f"   {MODELS_DIR}/")
    print()

    logger.debug("Waiting for user to complete Colab training...")
    input("Press Enter once you've completed training and placed the model file...")

    # Check if model exists
    model_path = MODELS_DIR / f"{model_name}.onnx"
    tflite_path = MODELS_DIR / f"{model_name}.tflite"

    logger.debug(f"Checking for model at: {model_path} or {tflite_path}")

    if not model_path.exists() and not tflite_path.exists():
        logger.warning("Model not found at expected locations")
        print(f"\nWarning: Model not found at {model_path}")
        print("Please ensure the model is placed in the correct location.")
        manual_path = input(
            "Enter the path to your model file (or press Enter to skip): "
        ).strip()
        if manual_path and Path(manual_path).exists():
            logger.info(f"User provided manual model path: {manual_path}")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            dest_path = MODELS_DIR / Path(manual_path).name
            shutil.copy(manual_path, dest_path)
            logger.info(f"Copied model to {dest_path}")
            print(f"Copied model to {MODELS_DIR}")
        else:
            logger.warning("No model path provided or path does not exist")
    else:
        found_path = model_path if model_path.exists() else tflite_path
        logger.info(f"Model found at: {found_path}")


def record_verifier_samples(
    model_name: str, venv_python: str
) -> tuple[list[Path], list[Path]]:
    """Record audio samples for custom verifier training."""
    logger.info("Starting verifier sample recording")
    print_header("Recording Verifier Samples")

    print("""
To improve accuracy and reduce false activations, we'll now record
samples of YOUR voice saying the wake word and some general speech.

Requirements:
- Quiet environment
- Speak naturally as you would when using the wake word
- Keep each recording brief (2-3 seconds for wake word, 10+ seconds for speech)
""")

    logger.debug("Waiting for user to be ready for recording...")
    input("Press Enter when ready to begin recording...")

    # Import audio libraries
    recording_script = '''
import sounddevice as sd
import soundfile as sf
import numpy as np
import sys
from pathlib import Path

def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone."""
    print(f"Recording for {duration} seconds...")
    print("Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

def save_audio(audio: np.ndarray, path: Path, sample_rate: int = 16000):
    """Save audio to WAV file."""
    sf.write(str(path), audio, sample_rate)

def main():
    import json
    args = json.loads(sys.argv[1])

    recordings_dir = Path(args["recordings_dir"])
    model_name = args["model_name"]
    mode = args["mode"]
    index = args.get("index", 0)
    duration = args.get("duration", 3.0)

    recordings_dir.mkdir(parents=True, exist_ok=True)

    audio = record_audio(duration)

    if mode == "positive":
        path = recordings_dir / f"{model_name}_positive_{index}.wav"
    else:
        path = recordings_dir / f"{model_name}_negative_{index}.wav"

    save_audio(audio, path)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
'''

    rec_script_path = SCRIPT_DIR / "_record_helper.py"
    logger.debug(f"Writing recording helper script to: {rec_script_path}")
    rec_script_path.write_text(recording_script)

    positive_clips = []
    negative_clips = []

    logger.debug(f"Creating recordings directory: {RECORDINGS_DIR}")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Record positive samples (wake word)
    logger.info("Recording positive (wake word) samples")
    print("\n--- Recording WAKE WORD samples ---")
    print("You'll record 5 samples of you saying the wake word.\n")

    for i in range(5):
        input(f"\nSample {i + 1}/5: Press Enter, then say the wake word...")
        logger.debug(f"Recording positive sample {i + 1}/5")
        args = json.dumps(
            {
                "recordings_dir": str(RECORDINGS_DIR),
                "model_name": model_name,
                "mode": "positive",
                "index": i,
                "duration": 3.0,
            }
        )
        run_cmd(
            [venv_python, str(rec_script_path), args],
            description=f"Record positive sample {i + 1}",
        )
        clip_path = RECORDINGS_DIR / f"{model_name}_positive_{i}.wav"
        positive_clips.append(clip_path)
        logger.debug(f"Recorded: {clip_path}")

        # Playback option
        play = input("Play back recording? [y/N]: ").strip().lower()
        if play == "y":
            run_cmd(
                ["afplay", str(positive_clips[-1])],
                check=False,
                description="Playback recording",
            )
            redo = input("Re-record this sample? [y/N]: ").strip().lower()
            if redo == "y":
                logger.debug(f"User chose to re-record sample {i + 1}")
                positive_clips.pop()
                continue

    # Record negative samples (general speech)
    logger.info("Recording negative (general speech) samples")
    print("\n--- Recording NEGATIVE samples ---")
    print("Now record 2 clips of general speech WITHOUT the wake word.")
    print("Talk about anything for 10-15 seconds each.\n")

    for i in range(2):
        input(
            f"\nSample {i + 1}/2: Press Enter, then speak naturally (no wake word)..."
        )
        logger.debug(f"Recording negative sample {i + 1}/2")
        args = json.dumps(
            {
                "recordings_dir": str(RECORDINGS_DIR),
                "model_name": model_name,
                "mode": "negative",
                "index": i,
                "duration": 15.0,
            }
        )
        run_cmd(
            [venv_python, str(rec_script_path), args],
            description=f"Record negative sample {i + 1}",
        )
        clip_path = RECORDINGS_DIR / f"{model_name}_negative_{i}.wav"
        negative_clips.append(clip_path)
        logger.debug(f"Recorded: {clip_path}")

    # Cleanup helper script
    logger.debug(f"Cleaning up helper script: {rec_script_path}")
    rec_script_path.unlink()

    logger.info(
        f"Recording complete: {len(positive_clips)} positive, {len(negative_clips)} negative samples"
    )
    print(
        f"\nRecorded {len(positive_clips)} positive and {len(negative_clips)} negative samples."
    )
    return positive_clips, negative_clips


def train_verifier(
    model_name: str,
    positive_clips: list[Path],
    negative_clips: list[Path],
    venv_python: str,
):
    """Train the custom verifier model."""
    logger.info("Starting verifier model training")
    print_header("Training Custom Verifier")

    # Find the base model
    logger.debug(f"Looking for base model in {MODELS_DIR}")
    model_path = None
    for ext in [".onnx", ".tflite"]:
        candidate = MODELS_DIR / f"{model_name}{ext}"
        logger.debug(f"Checking for: {candidate}")
        if candidate.exists():
            model_path = candidate
            logger.debug(f"Found base model: {model_path}")
            break

    if not model_path:
        logger.error(f"Base model not found in {MODELS_DIR}")
        print(f"Error: Base model not found in {MODELS_DIR}")
        print("Please ensure the base model training completed successfully.")
        return None

    verifier_script = """
import sys
import json
from pathlib import Path

def main():
    args = json.loads(sys.argv[1])

    positive_clips = [Path(p) for p in args["positive_clips"]]
    negative_clips = [Path(p) for p in args["negative_clips"]]
    output_path = Path(args["output_path"])
    model_name = args["model_name"]

    import openwakeword

    print(f"Training verifier with {len(positive_clips)} positive and {len(negative_clips)} negative clips...")

    openwakeword.train_custom_verifier(
        positive_reference_clips=[str(p) for p in positive_clips],
        negative_reference_clips=[str(p) for p in negative_clips],
        output_path=str(output_path),
        model_name=model_name,
    )

    print(f"Verifier saved to: {output_path}")

if __name__ == "__main__":
    main()
"""

    verifier_script_path = SCRIPT_DIR / "_verifier_helper.py"
    logger.debug(f"Writing verifier training script to: {verifier_script_path}")
    verifier_script_path.write_text(verifier_script)

    verifier_output = MODELS_DIR / f"{model_name}_verifier.pkl"

    logger.debug(f"Positive clips: {[str(p) for p in positive_clips]}")
    logger.debug(f"Negative clips: {[str(p) for p in negative_clips]}")
    logger.debug(f"Output path: {verifier_output}")
    logger.debug(f"Base model: {model_path}")

    args = json.dumps(
        {
            "positive_clips": [str(p) for p in positive_clips],
            "negative_clips": [str(p) for p in negative_clips],
            "output_path": str(verifier_output),
            "model_name": str(model_path),
        }
    )

    logger.info("Running verifier training...")
    run_cmd(
        [venv_python, str(verifier_script_path), args],
        description="Train verifier model",
    )

    # Cleanup
    logger.debug(f"Cleaning up verifier script: {verifier_script_path}")
    verifier_script_path.unlink()

    if verifier_output.exists():
        logger.info(f"Verifier model saved to: {verifier_output}")
    else:
        logger.error(f"Verifier model not found at expected path: {verifier_output}")

    return verifier_output


def generate_usage_example(model_name: str):
    """Generate example code for using the trained model."""
    logger.info("Generating usage example and integration instructions")
    print_header("Training Complete!")

    model_path = None
    for ext in [".onnx", ".tflite"]:
        candidate = MODELS_DIR / f"{model_name}{ext}"
        if candidate.exists():
            model_path = candidate
            logger.debug(f"Found model: {model_path}")
            break

    verifier_path = MODELS_DIR / f"{model_name}_verifier.pkl"

    logger.info(f"Model path: {model_path}")
    logger.info(f"Verifier path: {verifier_path} (exists: {verifier_path.exists()})")

    print("Your trained model files:")
    print(f"  Base model: {model_path}")
    if verifier_path.exists():
        print(f"  Verifier:   {verifier_path}")

    print("\n--- Example Usage ---\n")

    example_code = f'''
from openwakeword.model import Model

# Initialize with your custom model
model = Model(
    wakeword_models=["{model_path}"],
'''

    if verifier_path.exists():
        example_code += f'''    custom_verifier_models={{"{model_name}": "{verifier_path}"}},
    custom_verifier_threshold=0.3,
'''

    example_code += """)

# Process audio (16kHz, mono, int16 or float32)
import numpy as np
audio_chunk = np.zeros(1280, dtype=np.int16)  # 80ms at 16kHz

prediction = model.predict(audio_chunk)
for wake_word, score in prediction.items():
    if score > 0.5:
        print(f"Detected: {wake_word} (score: {score:.2f})")
"""

    print(example_code)

    # Save example to file
    example_path = MODELS_DIR / f"{model_name}_example.py"
    example_path.write_text(example_code)
    print(f"\nExample saved to: {example_path}")

    # Print integration hint
    print("\n--- Integration with Temporal ---\n")
    print("To use this model in the Olorin temporal component:")
    print(f"""
1. Update settings.json:

   "temporal": {{
     "wake_word": {{
       "engine": "openwakeword",
       "phrase": "{model_name.replace("_", " ")}"
     }},
     "openwakeword": {{
       "model_path": "{model_path}",
       "verifier_path": "{verifier_path if verifier_path.exists() else "null"}",
       "threshold": 0.5,
       "verifier_threshold": 0.3
     }}
   }}

2. Create the OpenWakeWord detector in temporal/openwakeword_detector.py
   (see previous instructions)
""")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a custom OpenWakeWord wake word model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./train.py                    # Interactive training with default settings
  ./train.py --debug            # Enable debug output to console
  ./train.py --skip-verifier    # Skip the verifier training step

Logs are always written to: tools/train-wake-word/logs/
        """,
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug output to console"
    )
    parser.add_argument(
        "--skip-verifier",
        action="store_true",
        help="Skip the verifier training step",
    )
    parser.add_argument(
        "--wake-word",
        "-w",
        type=str,
        help="Wake word/phrase (skips interactive prompt)",
    )
    args = parser.parse_args()

    # Initialize logging first
    setup_logging(debug=args.debug)
    logger.info("=" * 60)
    logger.info("OpenWakeWord Custom Wake Word Trainer started")
    logger.info("=" * 60)

    print_header("OpenWakeWord Custom Wake Word Trainer")

    try:
        # Step 1: Check pyenv
        print_step(1, "Checking pyenv installation")
        if not check_pyenv():
            logger.error("pyenv not found in PATH")
            print("Error: pyenv not found in PATH")
            print("Please install pyenv: https://github.com/pyenv/pyenv#installation")
            sys.exit(1)
        print("pyenv found.")

        # Step 2: Install Python 3.10
        print_step(2, f"Setting up Python {PYTHON_VERSION}")
        if not check_python_version_installed(PYTHON_VERSION):
            install_python_version(PYTHON_VERSION)
        else:
            logger.debug(f"Python {PYTHON_VERSION} already installed")
            print(f"Python {PYTHON_VERSION} already installed.")

        python_path = get_python_path(PYTHON_VERSION)
        print(f"Using Python: {python_path}")

        # Step 3: Create virtual environment
        print_step(3, "Creating virtual environment")
        venv_path = SCRIPT_DIR / VENV_NAME
        if not venv_path.exists():
            create_venv(python_path, venv_path)
        else:
            logger.debug(f"Virtual environment already exists at {venv_path}")
            print(f"Virtual environment already exists at {venv_path}")

        venv_python = get_venv_python(venv_path)

        # Step 4: Install dependencies
        print_step(4, "Installing dependencies")
        has_piper = install_dependencies(venv_python)

        # Step 5: Get wake word from user
        print_step(5, "Configuring wake word")
        if args.wake_word:
            wake_word = args.wake_word.strip().lower()
            model_name = wake_word.replace(" ", "_").replace("'", "")
            logger.info(f"Wake word from command line: '{wake_word}' -> {model_name}")
            print(f"Using wake word from command line: '{wake_word}'")
        else:
            wake_word, model_name = prompt_wake_word()

        # Generate training config
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = CONFIG_DIR / f"{model_name}.yaml"
        generate_training_config(wake_word, model_name, config_path)

        # Step 6: Train base model
        print_step(6, "Training base model")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        training_method = check_platform_support()
        logger.info(f"Platform: {sys.platform}, Training method: {training_method}")
        print(f"Platform: {sys.platform}, Training method: {training_method}")

        if training_method == "native" and has_piper:
            # Native Linux training
            logger.info("Training natively on Linux")
            print("Training natively on Linux...")
            run_cmd(
                [
                    venv_python,
                    "-m",
                    "openwakeword.train",
                    "--training_config",
                    str(config_path),
                    "--output_dir",
                    str(MODELS_DIR),
                ],
                check=False,
                description="Native Linux training",
            )
        elif training_method == "container":
            train_with_container(wake_word, model_name, venv_python)
        else:
            print_colab_instructions(wake_word, model_name)

        # Step 7: Record verifier samples
        print_step(7, "Recording verifier samples")

        skip_verifier = args.skip_verifier
        if not skip_verifier:
            skip_verifier = (
                input("Skip verifier training? [y/N]: ").strip().lower() == "y"
            )

        if skip_verifier:
            logger.info("Skipping verifier training (user choice)")
        else:
            positive_clips, negative_clips = record_verifier_samples(
                model_name, venv_python
            )

            # Step 8: Train verifier
            print_step(8, "Training verifier model")
            train_verifier(model_name, positive_clips, negative_clips, venv_python)

        # Step 9: Generate usage example
        print_step(9, "Generating usage example")
        generate_usage_example(model_name)

        logger.info("Training completed successfully!")
        print("\nDone! Your custom wake word model is ready to use.")
        print(f"Logs saved to: {LOG_DIR}/")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        print("\n\nTraining interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        print(f"\nError: {e}")
        print(f"Check logs for details: {LOG_DIR}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
