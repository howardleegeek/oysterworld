#!/usr/bin/env python3
"""
PhysicalFish Demo — Pre-Flight Check
Verifies all components are ready to run.
"""

import sys
import subprocess
import json
from pathlib import Path

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check(name: str, condition: bool, message: str = "") -> bool:
    """Print check result"""
    if condition:
        print(f"{GREEN}✓{RESET} {name}")
        if message:
            print(f"  {message}")
        return True
    else:
        print(f"{RED}✗{RESET} {name}")
        if message:
            print(f"  {RED}{message}{RESET}")
        return False


def main():
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  PhysicalFish Demo — Pre-Flight Check                    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")

    all_ok = True

    # Check Python
    print(f"{BLUE}Python Environment{RESET}")
    print("────────────────────────────────────────────────────────────")
    py_version = sys.version_info
    all_ok &= check(
        "Python 3.8+",
        py_version.major == 3 and py_version.minor >= 8,
        f"Version: {py_version.major}.{py_version.minor}.{py_version.micro}",
    )

    # Check numpy
    try:
        import numpy as np

        all_ok &= check("NumPy", True, f"Version: {np.__version__}")
    except ImportError:
        all_ok &= check("NumPy", False, "Install: pip install numpy")

    print()

    # Check project structure
    print(f"{BLUE}Project Structure{RESET}")
    print("────────────────────────────────────────────────────────────")

    script_dir = Path(__file__).parent

    required_files = [
        ("README.md", script_dir / "README.md"),
        ("run_demo.sh", script_dir / "run_demo.sh"),
        ("real-data/capture_real.py", script_dir / "real-data" / "capture_real.py"),
        (
            "verification/verify_physics.py",
            script_dir / "verification" / "verify_physics.py",
        ),
        (
            "autoresearch/optimize_loop.py",
            script_dir / "autoresearch" / "optimize_loop.py",
        ),
        (
            "synthetic-data/godot-project/project.godot",
            script_dir / "synthetic-data" / "godot-project" / "project.godot",
        ),
    ]

    for name, path in required_files:
        all_ok &= check(
            name, path.exists(), str(path) if path.exists() else f"Missing: {path}"
        )

    print()

    # Check Godot
    print(f"{BLUE}Godot Engine{RESET}")
    print("────────────────────────────────────────────────────────────")

    godot_ok = False
    try:
        result = subprocess.run(
            ["godot", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            # Check if version 4.x
            godot_ok = version.startswith("4.")
            all_ok &= check(
                "Godot 4.x",
                godot_ok,
                f"Version: {version}" if godot_ok else f"Version {version} (need 4.x)",
            )
        else:
            all_ok &= check("Godot 4.x", False, "Godot installed but returned error")
    except FileNotFoundError:
        all_ok &= check(
            "Godot 4.x", False, "Not found in PATH. Install: brew install godot"
        )
    except subprocess.TimeoutExpired:
        all_ok &= check("Godot 4.x", False, "Command timed out")
    except Exception as e:
        all_ok &= check("Godot 4.x", False, f"Error: {e}")

    print()

    # Test imports
    print(f"{BLUE}Component Tests{RESET}")
    print("────────────────────────────────────────────────────────────")

    # Test real data module
    try:
        sys.path.insert(0, str(script_dir / "real-data"))
        import capture_real

        all_ok &= check("Real data module", True)
    except Exception as e:
        all_ok &= check("Real data module", False, str(e))

    # Test verification module
    try:
        sys.path.insert(0, str(script_dir / "verification"))
        import verify_physics

        all_ok &= check("Verification module", True)
    except Exception as e:
        all_ok &= check("Verification module", False, str(e))

    # Test autoresearch module
    try:
        sys.path.insert(0, str(script_dir / "autoresearch"))
        import optimize_loop

        all_ok &= check("AutoResearch module", True)
    except Exception as e:
        all_ok &= check("AutoResearch module", False, str(e))

    print()

    # Summary
    print(f"{BLUE}Summary{RESET}")
    print("────────────────────────────────────────────────────────────")

    if all_ok:
        print(f"{GREEN}✓ All checks passed!{RESET}")
        print()
        print("You can now run the demo:")
        print(f"  {YELLOW}./run_demo.sh{RESET}")
        print()
        print("Or with custom parameters:")
        print(f"  {YELLOW}./run_demo.sh --target-score 0.95 --max-iterations 20{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some checks failed.{RESET}")
        print()
        print("Please fix the issues above before running the demo.")
        print()
        print("Common fixes:")
        print(f"  {YELLOW}pip install numpy{RESET}")
        print(f"  {YELLOW}brew install godot{RESET}  (macOS)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
