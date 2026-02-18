"""
Simple Google Colab Setup Script for Uni-GCR with Research HSTU
FIXED VERSION: Pins PyTorch 2.6.0 + FBGEMM 1.1.0 to resolve symbol errors.
"""

import os
import sys
import subprocess
import importlib

def run_command(cmd, description=""):
    """Run shell command and print status."""
    if description:
        print(f"▶️  {description}")
    print(f"   Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"❌ Failed with exit code {result.returncode}")
        return False
    return True

def check_runtime_restart_needed():
    """Checks if the loaded PyTorch version matches the installed one."""
    try:
        # Check what is installed on disk
        import pkg_resources
        installed_torch = pkg_resources.get_distribution("torch").version

        # Check what is currently loaded in memory
        import torch
        loaded_torch = torch.__version__

        if installed_torch != loaded_torch:
            print("\n⚠️  RUNTIME RESTART REQUIRED ⚠️")
            print(f"   Installed: {installed_torch}")
            print(f"   Loaded:    {loaded_torch}")
            print("   Colab pre-loaded the old version. You must restart the runtime to load the new one.")
            return True
    except:
        pass
    return False

def setup_colab_environment():
    """Setup complete environment for Uni-GCR in Colab."""

    print("=" * 70)
    print("Setting up Uni-GCR (Fixed for PyTorch 2.6.0 Compatibility)")
    print("=" * 70)

    # Step 1: Force Install Compatible Versions
    print("\n📦 Step 1: Installing Compatible PyTorch 2.6.0 & FBGEMM 1.1.0")

    # 1. Uninstall the default "Nightly" versions that cause the crash
    run_command(
        "pip uninstall -y torch torchvision torchaudio fbgemm-gpu torchrec",
        "Cleaning up incompatible nightly versions..."
    )

    # 2. Install PyTorch 2.6.0 (Stable)
    run_command(
        "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "Installing PyTorch 2.6.0 (Stable)..."
    )

    # 3. Install FBGEMM 1.1.0 (Must match PyTorch 2.6)
    run_command(
        "pip install fbgemm-gpu==1.1.0 --index-url https://download.pytorch.org/whl/cu124",
        "Installing FBGEMM 1.1.0..."
    )

    # 4. Install TorchRec 1.1.0 (Must match PyTorch 2.6)
    run_command(
        "pip install torchrec==1.1.0 --index-url https://download.pytorch.org/whl/cu124",
        "Installing TorchRec 1.1.0..."
    )

    # Step 2: Clone UniGCR repository
    print("\n📦 Step 2: Clone UniGCR Repository")
    if not os.path.exists('/content/UniGCR_New'):
        run_command(
            "git clone -b Zhirong https://github.com/kzhirong/UniGCR_New.git /content/UniGCR_New",
            "Cloning UniGCR..."
        )
    else:
        print("✅ UniGCR already cloned")
        run_command(
            "cd /content/UniGCR_New && git pull",
            "Updating UniGCR...",
        )

    # Step 5: Clean up deprecated utils package (fixes circular import)
    print("\n🧹 Step 5: Clean up deprecated files")
    deprecated_utils = "/content/UniGCR_New/src/utils"
    if os.path.exists(deprecated_utils):
        run_command(
            f"rm -rf {deprecated_utils}",
            "Removing deprecated utils package directory..."
        )
        print("✅ Removed src/utils/ directory (was causing circular import)")

    # Step 6: Install other dependencies
    print("\n📦 Step 6: Install other dependencies")
    run_command(
        "pip install numpy pandas scikit-learn tqdm iopath gin-config",
        "Installing Python packages (DeepSpeed not needed for single-GPU)..."
    )

    # Step 7: Clone and install generative_recommenders
    print("\n📦 Step 7: Install generative_recommenders")

    gen_rec_path = "/content/generative_recommenders"

    if not os.path.exists(gen_rec_path):
        run_command(
            f"git clone https://github.com/facebookresearch/generative-recommenders.git {gen_rec_path}",
            "Cloning generative_recommenders..."
        )
    else:
        print("✅ generative_recommenders already cloned")

    print("\n   Installing generative_recommenders...")
    # CRITICAL: Use --no-deps to prevent it from upgrading torch back to incompatible versions
    run_command(
        f"cd {gen_rec_path} && pip install --no-deps -e .",
        "Running pip install -e (Safe Mode)..."
    )

    # CHECK FOR RESTART
    if check_runtime_restart_needed():
        print("=" * 70)
        print("🛑 STOPPING SCRIPT: PLEASE RESTART RUNTIME")
        print("1. Go to 'Runtime' > 'Restart Session'")
        print("2. Run this script again (It will skip installs and go to verification)")
        print("=" * 70)
        return False

    # Step 8: Check fbgemm operations availability
    print("\n🔧 Step 8: Check fbgemm operations")

    import torch

    # Check version
    try:
        import fbgemm_gpu
        print(f"   fbgemm_gpu version: {fbgemm_gpu.__version__}")
    except Exception as e:
        print(f"   ⚠️  Cannot import fbgemm_gpu: {e}")
        return False

    # Check operations
    print("\n   Checking required operations:")
    required_ops = ['asynchronous_complete_cumsum', 'dense_to_jagged', 'jagged_to_padded_dense']
    missing_ops = []

    for op in required_ops:
        if hasattr(torch.ops.fbgemm, op):
            print(f"   ✅ {op}")
        else:
            print(f"   ❌ {op} - MISSING")
            missing_ops.append(op)

    if missing_ops:
        print(f"\n❌ {len(missing_ops)}/3 operations missing!")
        return False
    else:
        print("\n🎉 All required operations available!")

    # Step 9: Verify Research HSTU imports
    print("\n✅ Step 9: Verify Research HSTU imports")

    try:
        sys.path.insert(0, gen_rec_path)
        from generative_recommenders.research.modeling.sequential.hstu import HSTU
        print("✅ Research HSTU imports successfully!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary():
    """Print setup summary and next steps."""
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)

    print("\nNext Steps:")
    print("  %cd /content/UniGCR_New")
    print("  !python test_hstu_integration.py")

if __name__ == "__main__":
    success = setup_colab_environment()

    if success:
        print_summary()
    else:
        print("\n❌ Setup incomplete / Restart Required")