"""
Simple Google Colab Setup Script for Uni-GCR with Research HSTU

Just installs dependencies and checks what's available.
You can modify fbgemm installation as needed.
"""

import os
import sys
import subprocess

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

def setup_colab_environment():
    """Setup complete environment for Uni-GCR in Colab."""

    print("=" * 70)
    print("Setting up Uni-GCR with Research HSTU (Simple Installation)")
    print("=" * 70)

    # Step 1: Check PyTorch version
    print("\n🔥 Step 1: Check PyTorch and CUDA")
    import torch
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None

    print(f"✅ PyTorch: {pytorch_version}")
    print(f"✅ CUDA: {cuda_version}")

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

    # Step 3: Install fbgemm_gpu (SIMPLE - you can modify this)
    print("\n📦 Step 3: Install fbgemm_gpu")
    print("   You can modify this command to try different sources/versions")

    run_command(
        "pip install fbgemm-gpu",
        "Installing fbgemm-gpu from PyPI..."
    )

    # Step 4: Install torchrec
    print("\n📦 Step 4: Install torchrec")
    run_command(
        "pip install torchrec",
        "Installing torchrec from PyPI..."
    )

    # Step 5: Install other dependencies
    print("\n📦 Step 5: Install other dependencies")
    run_command(
        "pip install numpy pandas scikit-learn tqdm iopath gin-config",
        "Installing Python packages..."
    )

    # Step 6: Clone and install generative_recommenders
    print("\n📦 Step 6: Install generative_recommenders")

    gen_rec_path = "/content/generative_recommenders"

    if not os.path.exists(gen_rec_path):
        run_command(
            f"git clone https://github.com/facebookresearch/generative-recommenders.git {gen_rec_path}",
            "Cloning generative_recommenders..."
        )
    else:
        print("✅ generative_recommenders already cloned")

    print("\n   Installing generative_recommenders...")
    run_command(
        f"cd {gen_rec_path} && pip install -e .",
        "Running pip install -e ..."
    )

    # Step 7: Check fbgemm operations availability
    print("\n🔧 Step 7: Check fbgemm operations")

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
        print("\n   Debugging info:")
        print(f"   Available fbgemm operations: {[op for op in dir(torch.ops.fbgemm) if not op.startswith('_')][:20]}")
        print("\n   Try modifying Step 3 to install fbgemm from different source:")
        print("   - PyTorch wheel index: pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121")
        print("   - Specific version: pip install fbgemm-gpu==1.3.0")
        print("   - Nightly: pip install fbgemm-gpu-nightly")
        return False
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
        print("\n❌ Setup incomplete - see errors above")
        print("\n💡 Tips:")
        print("  1. Check fbgemm-gpu version compatibility with your PyTorch version")
        print("  2. Try different installation sources (see Step 3 output)")
        print("  3. Check CUDA version compatibility")
