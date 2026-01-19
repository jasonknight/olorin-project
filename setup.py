#!/usr/bin/env python3
"""
Olorin Project Setup Script

Checks for all required dependencies and provides installation instructions.

Usage:
    python3 setup.py           Interactive mode (prompts for missing deps)
    python3 setup.py --check   Non-interactive mode (just check, no prompts)
    python3 setup.py -c        Same as --check
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional

# Parse command line arguments early
_parser = argparse.ArgumentParser(
    description="Olorin Project Setup Script",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python3 setup.py           # Interactive setup
    python3 setup.py --check   # Just check dependencies, no prompts
""",
)
_parser.add_argument(
    "--check",
    "-c",
    action="store_true",
    help="Non-interactive mode (just check dependencies, no prompts)",
)
_args, _ = _parser.parse_known_args()

# Global flag for interactive mode
INTERACTIVE = not _args.check


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")


def print_subheader(text: str) -> None:
    """Print a subsection header."""
    print(f"\n{Colors.CYAN}{'-' * 40}{Colors.RESET}")
    print(f"{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * 40}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[MISSING]{Colors.RESET} {text}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {text}")


def run_command(cmd: list[str], capture: bool = True) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=30,
        )
        return result.returncode == 0, result.stdout.strip()
    except FileNotFoundError:
        return False, ""
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def get_version(cmd: str, args: list[str] | None = None) -> Optional[str]:
    """Get version string from a command."""
    if args is None:
        args = ["--version"]
    success, output = run_command([cmd] + args)
    if success and output:
        # Return first line of version output
        return output.split("\n")[0]
    return None


def ask_continue(install_cmd: str, required: bool = True) -> bool:
    """Ask user if they want to continue after showing install instructions.

    Args:
        install_cmd: The command to install the missing dependency
        required: If True, this is a required dependency; if False, optional

    Returns:
        True to continue, False to skip
    """
    print(f"\n{Colors.YELLOW}Installation command:{Colors.RESET}")
    print(f"  {Colors.BOLD}{install_cmd}{Colors.RESET}")
    print()

    if not INTERACTIVE:
        # Non-interactive mode: just report and continue
        if required:
            print_info("Run the command above to install, then re-run setup.py")
        return True

    while True:
        response = input(
            f"{Colors.BOLD}Continue after installing? (y/n/s=skip): {Colors.RESET}"
        ).lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            print("\nExiting setup. Please install the missing dependency and re-run.")
            sys.exit(1)
        elif response in ("s", "skip"):
            print_warning("Skipping this dependency...")
            return False
        else:
            print("Please enter 'y' to continue, 'n' to exit, or 's' to skip.")


def check_brew_package(package: str, install_name: str | None = None) -> bool:
    """Check if a Homebrew package is installed."""
    install_name = install_name or package
    success, _ = run_command(["brew", "list", package])
    return success


# =============================================================================
# Dependency Checkers
# =============================================================================


def check_python() -> bool:
    """Check Python version."""
    print_subheader("Python")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 11:
        print_success(f"Python {version_str} installed")
        return True
    else:
        print_error(f"Python {version_str} found, but 3.11+ required")
        return ask_continue("brew install python@3.11")


def check_rust() -> bool:
    """Check Rust toolchain."""
    print_subheader("Rust Toolchain")

    if check_command_exists("rustc"):
        version = get_version("rustc")
        if version:
            print_success(f"Rust compiler: {version}")
        else:
            print_success("Rust compiler installed")

        # Check cargo
        if check_command_exists("cargo"):
            version = get_version("cargo")
            if version:
                print_success(f"Cargo: {version}")
            else:
                print_success("Cargo installed")
        return True
    else:
        print_error("Rust toolchain not found")
        print_info(
            "Rust is required for building: tools/write, tools/chat, tools/settings, tools/olorin-inspector"
        )
        return ask_continue("brew install rust")


def check_podman() -> bool:
    """Check Podman container runtime."""
    print_subheader("Container Runtime (Podman)")

    if check_command_exists("podman"):
        version = get_version("podman")
        if version:
            print_success(f"Podman: {version}")
        else:
            print_success("Podman installed")

        # Check if podman machine is running
        success, output = run_command(
            ["podman", "machine", "list", "--format", "{{.Name}} {{.Running}}"]
        )
        if success and "true" in output.lower():
            print_success("Podman machine is running")
        else:
            print_warning(
                "Podman machine may not be running. Start with: podman machine start"
            )

        return True
    else:
        print_error("Podman not found")
        print_info("Podman is required for running Kafka and ChromaDB containers")
        return ask_continue(
            "brew install podman && podman machine init && podman machine start"
        )


def check_pandoc() -> bool:
    """Check Pandoc (optional)."""
    print_subheader("Pandoc (Optional - Document Conversion)")

    if check_command_exists("pandoc"):
        version = get_version("pandoc")
        if version:
            print_success(f"Pandoc: {version}")
        else:
            print_success("Pandoc installed")
        return True
    else:
        print_warning("Pandoc not found (optional)")
        print_info("Pandoc enables DOC/DOCX/ODT document conversion in hippocampus")
        if not INTERACTIVE:
            print_info("Install with: brew install pandoc")
            return True
        response = input(f"{Colors.BOLD}Install pandoc? (y/n): {Colors.RESET}").lower()
        if response in ("y", "yes"):
            return ask_continue("brew install pandoc", required=False)
        return True  # Optional, don't block


def check_ollama() -> bool:
    """Check Ollama (optional AI backend)."""
    print_subheader("Ollama (Optional - Local LLM Inference)")

    # Check various possible locations
    ollama_paths = [
        "ollama",
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        "/Applications/Ollama.app/Contents/Resources/ollama",
    ]

    for path in ollama_paths:
        if check_command_exists(path) or os.path.exists(path):
            version = get_version(path) if check_command_exists(path) else None
            if version:
                print_success(f"Ollama: {version}")
            else:
                print_success(f"Ollama found at: {path}")
            return True

    print_warning("Ollama not found (optional)")
    print_info("Ollama is one of the supported AI inference backends")
    print_info("Other options: Exo (distributed), Anthropic Claude (API key required)")
    if not INTERACTIVE:
        print_info("Install with: brew install ollama")
        return True
    response = input(f"{Colors.BOLD}Install Ollama? (y/n): {Colors.RESET}").lower()
    if response in ("y", "yes"):
        return ask_continue("brew install ollama", required=False)
    return True  # Optional


def check_precommit() -> bool:
    """Check pre-commit (optional for development)."""
    print_subheader("Pre-commit (Optional - Development)")

    if check_command_exists("pre-commit"):
        version = get_version("pre-commit")
        if version:
            print_success(f"Pre-commit: {version}")
        else:
            print_success("Pre-commit installed")
        return True
    else:
        print_warning("Pre-commit not found (optional)")
        print_info("Pre-commit is used for code quality hooks during development")
        if not INTERACTIVE:
            print_info("Install with: brew install pre-commit")
            return True
        response = input(
            f"{Colors.BOLD}Install pre-commit? (y/n): {Colors.RESET}"
        ).lower()
        if response in ("y", "yes"):
            return ask_continue("brew install pre-commit", required=False)
        return True  # Optional


def check_ffmpeg() -> bool:
    """Check ffmpeg (optional for audio processing)."""
    print_subheader("FFmpeg (Optional - Audio Processing)")

    if check_command_exists("ffmpeg"):
        version = get_version("ffmpeg")
        if version:
            print_success(f"FFmpeg: {version}")
        else:
            print_success("FFmpeg installed")
        return True
    else:
        print_warning("FFmpeg not found (optional)")
        print_info("FFmpeg may be needed for audio processing in TTS/STT components")
        if not INTERACTIVE:
            print_info("Install with: brew install ffmpeg")
            return True
        response = input(f"{Colors.BOLD}Install ffmpeg? (y/n): {Colors.RESET}").lower()
        if response in ("y", "yes"):
            return ask_continue("brew install ffmpeg", required=False)
        return True  # Optional


def check_python_venvs(project_root: Path) -> bool:
    """Check Python virtual environments for each component."""
    print_subheader("Python Virtual Environments")

    components = [
        ("broca", "Text-to-Speech"),
        ("cortex", "AI Processing"),
        ("hippocampus", "Document Ingestion"),
        ("temporal", "Voice Input"),
        ("tools/embeddings", "Embeddings Tool Server"),
        ("tools/search", "Search Tool Server"),
    ]

    all_ok = True
    missing = []

    for component, description in components:
        venv_path = project_root / component / "venv"
        req_path = project_root / component / "requirements.txt"

        if venv_path.exists():
            # Check if venv has python
            python_path = venv_path / "bin" / "python"
            if python_path.exists():
                print_success(f"{component}/venv exists ({description})")
            else:
                print_warning(f"{component}/venv exists but may be incomplete")
        else:
            if req_path.exists():
                print_error(f"{component}/venv missing ({description})")
                missing.append((component, req_path))
                all_ok = False
            else:
                print_info(f"{component} has no requirements.txt, skipping venv check")

    if missing:
        print()
        print_info("To create missing virtual environments:")
        for component, req_path in missing:
            print(
                f"  cd {component} && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
            )
        print()
        if INTERACTIVE:
            response = input(f"{Colors.BOLD}Continue? (y/n): {Colors.RESET}").lower()
            if response not in ("y", "yes"):
                sys.exit(1)

    return all_ok


def check_rust_projects(project_root: Path) -> bool:
    """Check Rust project build status."""
    print_subheader("Rust Projects")

    rust_projects = [
        ("libs/config-rs", "Configuration Library"),
        ("libs/state-rs", "State Management Library"),
        ("tools/write", "File Writer Tool Server"),
        ("tools/chat", "TUI Chat Client"),
        ("tools/settings", "Settings TUI Editor"),
        ("tools/olorin-inspector", "Database Inspector"),
    ]

    all_ok = True
    needs_build = []

    for project, description in rust_projects:
        cargo_toml = project_root / project / "Cargo.toml"
        target_dir = project_root / project / "target"

        if cargo_toml.exists():
            # Check if there's a built binary
            release_dir = target_dir / "release"
            debug_dir = target_dir / "debug"

            if release_dir.exists() or debug_dir.exists():
                print_success(f"{project} built ({description})")
            else:
                print_warning(f"{project} not built yet ({description})")
                needs_build.append(project)
        else:
            print_info(f"{project} Cargo.toml not found, skipping")

    if needs_build:
        print()
        print_info("To build Rust projects:")
        for project in needs_build:
            print(f"  cargo build --release --manifest-path {project}/Cargo.toml")
        print()
        if INTERACTIVE:
            response = input(
                f"{Colors.BOLD}Continue without building? (y/n): {Colors.RESET}"
            ).lower()
            if response not in ("y", "yes"):
                sys.exit(1)

    return all_ok


def check_containers() -> bool:
    """Check container status."""
    print_subheader("Container Services")

    if not check_command_exists("podman"):
        print_warning("Podman not installed, skipping container checks")
        return True

    # Check if containers are running
    containers = [
        ("kafkaserver", "Kafka Message Broker", "9092"),
        ("chromadb", "ChromaDB Vector Database", "8000"),
    ]

    all_ok = True
    for name, description, port in containers:
        success, output = run_command(
            ["podman", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"]
        )
        if success and name in output:
            print_success(f"{name} container running ({description})")
        else:
            print_info(f"{name} container not running ({description})")
            print_info("  Will be started automatically by ./up script")

    return all_ok


def check_env_file(project_root: Path) -> bool:
    """Check .env file and API keys."""
    print_subheader("Environment Configuration")

    env_path = project_root / ".env"
    settings_path = project_root / "settings.json"

    # Check settings.json
    if settings_path.exists():
        print_success("settings.json exists")
    else:
        print_warning("settings.json not found - will use defaults")

    # Check .env
    if env_path.exists():
        print_success(".env file exists")

        # Check for Anthropic key
        with open(env_path) as f:
            content = f.read()

        if "ANTHROPIC_API_KEY" in content:
            # Check if it has a value
            for line in content.split("\n"):
                if line.startswith("ANTHROPIC_API_KEY="):
                    value = line.split("=", 1)[1].strip()
                    if value and value.startswith("sk-ant-"):
                        print_success("ANTHROPIC_API_KEY configured")
                    elif value:
                        print_warning(
                            "ANTHROPIC_API_KEY set but may be invalid (should start with sk-ant-)"
                        )
                    else:
                        print_warning("ANTHROPIC_API_KEY is empty")
                    break
        else:
            print_info("ANTHROPIC_API_KEY not set (only needed for Anthropic backend)")
    else:
        print_info(".env file not found")
        print_info("Create .env if you need to use Anthropic Claude API:")
        print_info("  echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env")

    return True


def check_directories(project_root: Path) -> bool:
    """Check required directories exist."""
    print_subheader("Directory Structure")

    directories = [
        ("data", "Shared runtime data"),
        ("logs", "Component logs"),
        (".pids", "Process ID tracking"),
        (Path.home() / "Documents" / "AI_IN", "Document input directory"),
    ]

    for dir_path, description in directories:
        if isinstance(dir_path, Path):
            full_path = dir_path
        else:
            full_path = project_root / dir_path

        if full_path.exists():
            print_success(f"{full_path} exists ({description})")
        else:
            print_info(f"{full_path} will be created ({description})")
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print_success(f"Created {full_path}")
            except Exception as e:
                print_warning(f"Could not create {full_path}: {e}")

    return True


def check_wakeword_model(project_root: Path) -> bool:
    """Check for wake word model (Picovoice Porcupine)."""
    print_subheader("Wake Word Model (Optional)")

    # Common locations for wake word models
    model_paths = [
        Path.home()
        / "Downloads"
        / "Hey-Computer_en_mac_v4_0_0"
        / "Hey-Computer_en_mac_v4_0_0.ppn",
        project_root / "temporal" / "data" / "models" / "wakeword.ppn",
    ]

    # Also check settings.json for configured path
    settings_path = project_root / "settings.json"
    if settings_path.exists():
        try:
            import json

            with open(settings_path) as f:
                settings = json.load(f)
            model_path = (
                settings.get("temporal", {}).get("porcupine", {}).get("model_path")
            )
            if model_path:
                model_paths.insert(0, Path(model_path).expanduser())
        except Exception:
            pass

    for path in model_paths:
        if path.exists():
            print_success(f"Wake word model found: {path}")
            return True

    print_warning("No wake word model (.ppn) found (optional)")
    print_info("Wake word detection requires a Picovoice Porcupine model")
    print_info("You can create one at: https://console.picovoice.ai/")
    print_info("Or the system will fall back to continuous listening mode")
    return True  # Optional


def check_whisper_models(project_root: Path) -> bool:
    """Check for Whisper speech-to-text models."""
    print_subheader("Speech-to-Text Models (Optional)")

    # Faster-whisper caches models in huggingface cache
    cache_paths = [
        Path.home() / ".cache" / "huggingface" / "hub",
        project_root / "temporal" / "data" / "models",
    ]

    found_models = False
    for cache_path in cache_paths:
        if cache_path.exists():
            # Look for whisper model directories
            for item in cache_path.iterdir():
                if (
                    "whisper" in item.name.lower()
                    or "faster-whisper" in str(item).lower()
                ):
                    print_success(f"Whisper model cache found: {item.name}")
                    found_models = True

    if not found_models:
        print_info("No cached Whisper models found")
        print_info("Models will be downloaded automatically on first use")
        print_info("Default model: 'small' (~500MB)")

    return True


def check_tts_models() -> bool:
    """Check for TTS models."""
    print_subheader("Text-to-Speech Models (Optional)")

    # Coqui TTS model cache locations
    cache_paths = [
        Path.home() / ".local" / "share" / "tts",
        Path.home() / ".cache" / "TTS",
    ]

    found_models = False
    for cache_path in cache_paths:
        if cache_path.exists():
            print_success(f"TTS model cache found: {cache_path}")
            found_models = True

    if not found_models:
        print_info("No cached TTS models found")
        print_info("Models will be downloaded automatically on first use")
        print_info("Default model: 'tts_models/en/vctk/vits' (~300MB)")

    return True


def check_embedding_models() -> bool:
    """Check for sentence-transformer embedding models."""
    print_subheader("Embedding Models (Optional)")

    cache_path = Path.home() / ".cache" / "huggingface" / "hub"

    if cache_path.exists():
        found_embedding = False
        for item in cache_path.iterdir():
            name_lower = item.name.lower()
            if (
                "nomic" in name_lower
                or "sentence-transformers" in name_lower
                or "minilm" in name_lower
            ):
                print_success(f"Embedding model found: {item.name}")
                found_embedding = True

        if not found_embedding:
            print_info("No embedding models found in cache")
            print_info("Models will be downloaded automatically on first use")
    else:
        print_info("Huggingface cache not found")
        print_info("Embedding models will be downloaded on first use")
        print_info("Default: nomic-ai/nomic-embed-text-v1.5 (~420MB)")

    return True


def check_ai_backends() -> bool:
    """Check available AI inference backends."""
    print_subheader("AI Inference Backends")

    backends_found = 0

    # Check Ollama
    if check_command_exists("ollama"):
        # Try to connect
        success, output = run_command(["curl", "-s", "http://localhost:11434/api/tags"])
        if success and output:
            print_success("Ollama running and accessible at localhost:11434")
            backends_found += 1
        else:
            print_info("Ollama installed but not running (start with: ollama serve)")
    else:
        print_info("Ollama not installed")

    # Check Exo
    success, output = run_command(["curl", "-s", "http://localhost:52415/v1/models"])
    if success and output:
        print_success("Exo running and accessible at localhost:52415")
        backends_found += 1
    else:
        print_info("Exo not running at localhost:52415")

    # Check Anthropic (by env key)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key.startswith("sk-ant-"):
        print_success("Anthropic API key found in environment")
        backends_found += 1
    else:
        # Check .env file
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            with open(env_path) as f:
                if "ANTHROPIC_API_KEY=sk-ant-" in f.read():
                    print_success("Anthropic API key configured in .env")
                    backends_found += 1
                else:
                    print_info("Anthropic API key not configured")
        else:
            print_info("Anthropic API key not configured")

    if backends_found == 0:
        print()
        print_warning("No AI inference backend is currently running/configured!")
        print_info("You need at least one of:")
        print_info("  1. Ollama: brew install ollama && ollama serve")
        print_info("  2. Exo: Start Exo on port 52415")
        print_info("  3. Anthropic: Add ANTHROPIC_API_KEY=sk-ant-... to .env")
        print()
        if INTERACTIVE:
            response = input(
                f"{Colors.BOLD}Continue anyway? (y/n): {Colors.RESET}"
            ).lower()
            if response not in ("y", "yes"):
                sys.exit(1)

    return True


def print_summary(project_root: Path) -> None:
    """Print setup summary and next steps."""
    print_header("Setup Summary")

    print(f"""
{Colors.GREEN}Setup check complete!{Colors.RESET}

{Colors.BOLD}Next Steps:{Colors.RESET}

1. {Colors.CYAN}Start the system:{Colors.RESET}
   ./up

2. {Colors.CYAN}Check status:{Colors.RESET}
   ./status

3. {Colors.CYAN}Start the chat client:{Colors.RESET}
   cd tools/chat && cargo run --release

4. {Colors.CYAN}View logs:{Colors.RESET}
   tail -f logs/*.log

{Colors.BOLD}Optional - Pull an Ollama model:{Colors.RESET}
   ollama pull llama3.2:3b

{Colors.BOLD}Optional - Build Rust tools:{Colors.RESET}
   cargo build --release --manifest-path tools/chat/Cargo.toml
   cargo build --release --manifest-path tools/write/Cargo.toml

{Colors.BOLD}Configuration:{Colors.RESET}
   Edit settings.json to customize the system
   Edit .env to add API keys (like ANTHROPIC_API_KEY)

{Colors.BOLD}Documentation:{Colors.RESET}
   See CLAUDE.md for full documentation
""")


def main() -> None:
    """Main setup function."""
    print(f"""
{Colors.BOLD}{Colors.BLUE}
 ██████╗ ██╗      ██████╗ ██████╗ ██╗███╗   ██╗
██╔═══██╗██║     ██╔═══██╗██╔══██╗██║████╗  ██║
██║   ██║██║     ██║   ██║██████╔╝██║██╔██╗ ██║
██║   ██║██║     ██║   ██║██╔══██╗██║██║╚██╗██║
╚██████╔╝███████╗╚██████╔╝██║  ██║██║██║ ╚████║
 ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
{Colors.RESET}
{Colors.BOLD}Olorin Project Setup{Colors.RESET}
Distributed AI Pipeline System
""")

    if not INTERACTIVE:
        print(
            f"{Colors.YELLOW}Running in check mode (--check). No prompts will be shown.{Colors.RESET}"
        )

    # Determine project root
    project_root = Path(__file__).parent.resolve()
    print(f"Project root: {project_root}")

    # Run all checks
    print_header("System Dependencies (Required)")
    check_python()
    check_rust()
    check_podman()

    print_header("System Dependencies (Optional)")
    check_pandoc()
    check_ollama()
    check_ffmpeg()
    check_precommit()

    print_header("Project Structure")
    check_directories(project_root)
    check_env_file(project_root)
    check_python_venvs(project_root)
    check_rust_projects(project_root)

    print_header("Container Services")
    check_containers()

    print_header("AI/ML Models")
    check_wakeword_model(project_root)
    check_whisper_models(project_root)
    check_tts_models()
    check_embedding_models()

    print_header("AI Inference Backends")
    check_ai_backends()

    # Print summary
    print_summary(project_root)


if __name__ == "__main__":
    main()
