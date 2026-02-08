"""
Google Colab Setup Script for Uni-GCR with Research HSTU (V3 - Fixed Compatibility)

This version handles PyTorch/fbgemm version compatibility issues.
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

def setup_colab_environment():
    """Setup complete environment for Uni-GCR in Colab."""

    print("=" * 70)
    print("Setting up Uni-GCR with Research HSTU (V3 - Compatibility Fixed)")
    print("=" * 70)

    # Step 1: Check PyTorch version first
    print("\n🔥 Step 1: Check PyTorch and CUDA")
    import torch
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None

    print(f"✅ PyTorch: {pytorch_version}")
    print(f"✅ CUDA: {cuda_version}")

    # Determine compatible fbgemm version based on PyTorch version
    print("\n📊 Checking compatibility...")

    pytorch_major_minor = '.'.join(pytorch_version.split('.')[:2])  # e.g., "2.9"

    # PyTorch 2.9 is too new - need to use PyTorch 2.4 or 2.5 for stable fbgemm
    if pytorch_major_minor >= "2.9":
        print("⚠️  PyTorch 2.9+ detected - this is very new and may lack stable fbgemm_gpu builds")
        print("   We'll try to install compatible versions...")

        # Downgrade to PyTorch 2.4 (stable, well-supported)
        print("\n📦 Downgrading to PyTorch 2.4 for compatibility...")
        run_command(
            "pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121",
            "Installing PyTorch 2.4.0 with CUDA 12.1..."
        )

        # Reload torch after reinstall
        import importlib
        importlib.reload(torch)
        print(f"   New PyTorch version: {torch.__version__}")
        cuda_version = "12.1"  # We just installed cu121

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
            allow_fail=True
        )

    # Step 3: Install fbgemm_gpu (with correct version for PyTorch 2.4)
    print("\n📦 Step 3: Install fbgemm_gpu (compatible version)")

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

    # Step 4: Install torchrec
    print("\n📦 Step 4: Install torchrec")
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

    # Step 5: Install other dependencies
    print("\n📦 Step 5: Install other dependencies")
    run_command(
        "pip install numpy pandas scikit-learn tqdm iopath gin-config --no-cache-dir",
        "Installing Python packages..."
    )

    # Step 6: Clone and install generative_recommenders (NO --quiet to see errors!)
    print("\n📦 Step 6: Install generative_recommenders")

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

    # Step 7: Verify fbgemm operations
    print("\n🔧 Step 7: Verify fbgemm operations")

    # Force reload
    import importlib
    for mod in ['fbgemm_gpu', 'torch']:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])

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

    # Step 8: Verify Research HSTU imports
    print("\n✅ Step 8: Verify Research HSTU imports")

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
    success = setup_colab_environment()

    if success:
        print_summary()
    else:
        print("\n❌ Setup incomplete - see errors above")
        print("   Common fixes:")
        print("   • Runtime → Restart runtime, then re-run")
        print("   • Try: pip install generative-recommenders manually")
