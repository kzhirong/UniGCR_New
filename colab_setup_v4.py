"""
Google Colab Setup Script for Uni-GCR with Research HSTU (V4 - Two-Phase Setup)

IMPORTANT: This script may require a runtime restart. Follow the prompts.

Phase 1: Check PyTorch version and install compatible version if needed
Phase 2: Install all dependencies after PyTorch is compatible
"""

import os
import sys
import subprocess

def run_command(cmd, description="", allow_fail=False):
    """Run shell command and print status."""
    if description:
        print(f"▶️  {description}")
    print(f"   Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Show output regardless of success/fail
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print("STDERR:", result.stderr)

    if result.returncode != 0 and not allow_fail:
        print(f"❌ Failed with exit code {result.returncode}")
        return False
    return True

def phase1_check_pytorch():
    """Phase 1: Check PyTorch and install compatible version if needed."""
    print("=" * 70)
    print("PHASE 1: PyTorch Compatibility Check")
    print("=" * 70)

    # Check current PyTorch version
    result = subprocess.run(
        "python -c 'import torch; print(torch.__version__)'",
        shell=True, capture_output=True, text=True
    )

    if result.returncode != 0:
        print("❌ PyTorch not found - installing PyTorch 2.4.0...")
        return install_pytorch_2_4()

    pytorch_version = result.stdout.strip()
    print(f"Current PyTorch version: {pytorch_version}")

    pytorch_major_minor = '.'.join(pytorch_version.split('.')[:2])

    if pytorch_major_minor >= "2.9":
        print(f"\n⚠️  PyTorch {pytorch_version} is too new for stable fbgemm-gpu")
        print("   Need to downgrade to PyTorch 2.4.0 for compatibility")
        return install_pytorch_2_4()
    elif pytorch_major_minor < "2.4":
        print(f"\n⚠️  PyTorch {pytorch_version} is too old")
        print("   Need to upgrade to PyTorch 2.4.0")
        return install_pytorch_2_4()
    else:
        print(f"✅ PyTorch {pytorch_version} is compatible")
        return "CONTINUE"

def install_pytorch_2_4():
    """Install PyTorch 2.4.0 and request runtime restart."""
    print("\n📦 Installing PyTorch 2.4.0 with CUDA 12.1...")

    success = run_command(
        "pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121",
        "Installing PyTorch 2.4.0..."
    )

    if not success:
        print("\n❌ PyTorch installation failed")
        return "FAILED"

    print("\n" + "=" * 70)
    print("✅ PyTorch 2.4.0 installed successfully!")
    print("=" * 70)
    print("\n🔄 RUNTIME RESTART REQUIRED")
    print("\nNext steps:")
    print("  1. Go to Runtime → Restart runtime (or Ctrl+M .)")
    print("  2. After restart, run this script again")
    print("  3. The script will detect PyTorch 2.4.0 and continue with Phase 2")
    print("\n" + "=" * 70)

    return "RESTART_NEEDED"

def phase2_install_dependencies():
    """Phase 2: Install all other dependencies."""
    print("\n" + "=" * 70)
    print("PHASE 2: Installing Dependencies")
    print("=" * 70)

    # Step 1: Clone UniGCR repository
    print("\n📦 Step 1: Clone UniGCR Repository")
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
            allow_fail=True
        )

    # Step 2: Install fbgemm_gpu
    print("\n📦 Step 2: Install fbgemm_gpu (compatible version)")

    # Uninstall any existing version
    run_command(
        "pip uninstall fbgemm-gpu fbgemm-gpu-nightly -y",
        "Removing old fbgemm-gpu...",
        allow_fail=True
    )

    # Install from PyTorch wheel index (matches our PyTorch version)
    run_command(
        "pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir",
        "Installing fbgemm-gpu from PyTorch wheel index..."
    )

    # Step 3: Install torchrec
    print("\n📦 Step 3: Install torchrec")
    run_command(
        "pip install torchrec --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir",
        "Installing torchrec from PyTorch wheel index...",
        allow_fail=True
    )

    # Fallback to PyPI if wheel index fails
    if not run_command("python -c 'import torchrec'", "Verifying torchrec...", allow_fail=True):
        print("   PyTorch wheel index failed, trying PyPI...")
        run_command(
            "pip install torchrec --no-cache-dir",
            "Installing torchrec from PyPI..."
        )

    # Step 4: Install other dependencies
    print("\n📦 Step 4: Install other dependencies")
    run_command(
        "pip install numpy pandas scikit-learn tqdm iopath gin-config --no-cache-dir",
        "Installing Python packages..."
    )

    # Step 5: Clone and install generative_recommenders
    print("\n📦 Step 5: Install generative_recommenders")

    gen_rec_path = "/content/generative_recommenders"

    if not os.path.exists(gen_rec_path):
        run_command(
            f"git clone https://github.com/facebookresearch/generative-recommenders.git {gen_rec_path}",
            "Cloning generative_recommenders..."
        )
    else:
        print("✅ generative_recommenders already cloned")

    # Install WITHOUT --quiet so we can see what goes wrong
    print("\n   Installing generative_recommenders (this may show warnings)...")
    success = run_command(
        f"cd {gen_rec_path} && pip install -e .",
        "Running pip install -e ...",
        allow_fail=True
    )

    if not success:
        print("⚠️  Installation had errors but continuing...")

    # Step 6: Verify fbgemm operations
    print("\n🔧 Step 6: Verify fbgemm operations")

    import torch

    # Check version
    try:
        import fbgemm_gpu
        print(f"   fbgemm_gpu version: {fbgemm_gpu.__version__}")
    except Exception as e:
        print(f"   ⚠️  Cannot determine fbgemm_gpu version: {e}")

    # Check operations
    print("\n   Checking required operations:")
    missing_ops = []
    required_ops = ['asynchronous_complete_cumsum', 'dense_to_jagged', 'jagged_to_padded_dense']

    for op in required_ops:
        if hasattr(torch.ops.fbgemm, op):
            print(f"   ✅ {op}")
        else:
            print(f"   ❌ {op}")
            missing_ops.append(op)

    if missing_ops:
        print(f"\n⚠️  {len(missing_ops)} operations missing - will use fallbacks")
        print("   Available fbgemm ops:")
        available = [op for op in dir(torch.ops.fbgemm) if not op.startswith('_')]
        print(f"   {', '.join(available[:15])}")
    else:
        print("\n🎉 All required operations available!")

    # Step 7: Verify Research HSTU imports
    print("\n✅ Step 7: Verify Research HSTU imports")

    try:
        sys.path.insert(0, gen_rec_path)
        from generative_recommenders.research.modeling.sequential.hstu import HSTU
        print("✅ Research HSTU imports successfully!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()

        # Try to diagnose
        print("\n🔍 Diagnosing...")
        print(f"   gen_rec_path exists: {os.path.exists(gen_rec_path)}")
        if os.path.exists(gen_rec_path):
            research_path = os.path.join(gen_rec_path, 'generative_recommenders', 'research')
            print(f"   research folder exists: {os.path.exists(research_path)}")
            if os.path.exists(research_path):
                print(f"   Contents: {os.listdir(research_path)[:10]}")

        return False

def print_summary():
    """Print setup summary and next steps."""
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)

    print("\nNext Steps:")
    print("  %cd /content/UniGCR_New")
    print("  !python test_hstu_integration.py")

    print("\nNote:")
    print("  • If fbgemm operations are missing, fallbacks will be used")
    print("  • Fallbacks are 10-30% slower but functionally correct")
    print("  • This is normal for some environments")

if __name__ == "__main__":
    # Phase 1: Check PyTorch compatibility
    phase1_result = phase1_check_pytorch()

    if phase1_result == "RESTART_NEEDED":
        print("\n⏸️  Pausing here - restart runtime and re-run this script")
        sys.exit(0)
    elif phase1_result == "FAILED":
        print("\n❌ Setup failed in Phase 1")
        sys.exit(1)
    elif phase1_result == "CONTINUE":
        # Phase 2: Install all dependencies
        success = phase2_install_dependencies()

        if success:
            print_summary()
        else:
            print("\n❌ Setup incomplete - see errors above")
            print("   Common fixes:")
            print("   • Try: pip install generative-recommenders manually")
