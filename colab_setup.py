"""
Google Colab Setup Script for Uni-GCR with Research HSTU

Run this in a Colab cell before testing.
"""

import os
import sys
import subprocess
import importlib

def run_command(cmd, description=""):
    """Run shell command and print status."""
    if description:
        print(f"▶️  {description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {cmd}")
        print(f"Error: {result.stderr}")
        return False
    return True

def setup_colab_environment():
    """Setup complete environment for Uni-GCR in Colab."""

    print("=" * 70)
    print("Setting up Uni-GCR Environment in Google Colab")
    print("=" * 70)

    # Step 1: Clone the repository
    print("\n📦 Step 1: Clone UniGCR Repository")
    if not os.path.exists('/content/UniGCR_New'):
        run_command(
            "git clone -b Zhirong https://github.com/kzhirong/UniGCR_New.git",
            "Cloning repository..."
        )
        os.chdir('/content/UniGCR_New')
    else:
        print("✅ Repository already cloned")
        os.chdir('/content/UniGCR_New')

    # Step 2: Check PyTorch and CUDA
    print("\n🔥 Step 2: Check PyTorch and CUDA")
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")

    # Step 3: Install fbgemm_gpu (required by generative_recommenders)
    print("\n📦 Step 3: Install fbgemm_gpu")
    print("⚠️  This may take 5-10 minutes...")

    # Determine CUDA version for fbgemm
    cuda_version = torch.version.cuda
    if cuda_version:
        cuda_major = cuda_version.split('.')[0]
        fbgemm_package = f"fbgemm-gpu==1.0.0"  # Use version compatible with PyTorch

        success = run_command(
            f"pip install {fbgemm_package} --no-cache-dir",
            f"Installing fbgemm_gpu for CUDA {cuda_version}..."
        )
        if not success:
            print("⚠️  fbgemm_gpu installation failed, trying CPU version...")
            run_command(
                "pip install fbgemm-gpu-cpu --no-cache-dir",
                "Installing CPU version as fallback..."
            )
    else:
        print("No CUDA detected, installing CPU version")
        run_command(
            "pip install fbgemm-gpu-cpu --no-cache-dir",
            "Installing fbgemm_gpu (CPU)..."
        )

    # Step 4: Install torchrec (required by generative_recommenders)
    print("\n📦 Step 4: Install torchrec")
    run_command(
        "pip install torchrec --no-cache-dir",
        "Installing torchrec..."
    )

    # Step 5: Install other dependencies
    print("\n📦 Step 5: Install other dependencies")
    dependencies = [
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
        "iopath",  # Required by generative_recommenders
    ]

    for dep in dependencies:
        dep_name = dep.split('>=')[0] if '>=' in dep else dep
        run_command(f"pip install '{dep}' --no-cache-dir", f"Installing {dep_name}...")

    # Step 6: Install generative_recommenders
    print("\n📦 Step 6: Install generative_recommenders")
    print("⚠️  This may take 5-10 minutes...")

    # Clone and install locally (more reliable than pip install from git)
    gen_rec_path = "/content/generative_recommenders"

    if not os.path.exists(gen_rec_path):
        print("▶️  Cloning generative_recommenders repository...")
        success = run_command(
            f"git clone https://github.com/facebookresearch/generative-recommenders.git {gen_rec_path}",
            "Cloning repository..."
        )
        if not success:
            print("❌ Failed to clone generative_recommenders")
            return False
    else:
        print("✅ Repository already cloned")

    # Install in editable mode
    print("▶️  Installing in editable mode...")
    original_dir = os.getcwd()
    os.chdir(gen_rec_path)

    success = run_command(
        "pip install -e . --no-cache-dir",
        "Running pip install -e ..."
    )

    os.chdir(original_dir)

    if success:
        print("✅ Installed successfully")
        # Force Python to recognize the new package
        import site
        import importlib
        importlib.invalidate_caches()
        site.main()
    else:
        print("❌ Installation failed")

    # Step 6.5: Patch generative_recommenders for compatibility (optional)
    print("\n🔧 Step 6.5: Patch for compatibility (optional)")

    # Patch: Make hammer imports optional (Research HSTU might use it internally)
    hstu_attention_file = f"{gen_rec_path}/generative_recommenders/ops/hstu_attention.py"

    if os.path.exists(hstu_attention_file):
        with open(hstu_attention_file, 'r') as f:
            content = f.read()

        # Check if already patched
        if "# PATCHED: Optional hammer import" not in content:
            print("▶️  Patching hstu_attention.py to make hammer imports optional...")

            # Replace the unconditional hammer import with conditional one
            original_import = "from hammer.v2.ops.triton.template.tlx_bw_hstu_attention import tlx_bw_hstu_mha_wrapper"

            patched_import = """# PATCHED: Optional hammer import
try:
    from hammer.v2.ops.triton.template.tlx_bw_hstu_attention import tlx_bw_hstu_mha_wrapper
except ImportError:
    # Fallback: create a dummy wrapper that uses PyTorch implementation
    def tlx_bw_hstu_mha_wrapper(*args, **kwargs):
        return None  # Will use PyTorch fallback
    print("⚠️  hammer.v2 not available, using PyTorch fallback for HSTU attention")"""

            if original_import in content:
                content = content.replace(original_import, patched_import)

                # Write patched content back
                with open(hstu_attention_file, 'w') as f:
                    f.write(content)

                print("✅ Successfully patched hstu_attention.py")
            else:
                print("⚠️  Import line not found (may be already patched or version mismatch)")
        else:
            print("✅ hstu_attention.py already patched")
    else:
        print(f"⚠️  File not found: {hstu_attention_file} (not critical for Research HSTU)")

    # Step 7: Patch fbgemm compatibility issues
    print("\n🔧 Step 7: Patch fbgemm compatibility")

    # Patch missing asynchronous_complete_cumsum
    import torch
    if not hasattr(torch.ops.fbgemm, 'asynchronous_complete_cumsum'):
        print("⚠️  fbgemm.asynchronous_complete_cumsum not found, adding fallback...")

        def async_cumsum_fallback(lengths):
            """Fallback implementation using torch.cumsum"""
            return torch.cat([
                torch.zeros(1, dtype=lengths.dtype, device=lengths.device),
                torch.cumsum(lengths, dim=0)
            ])

        torch.ops.fbgemm.asynchronous_complete_cumsum = async_cumsum_fallback
        print("✅ Registered fallback for asynchronous_complete_cumsum")
    else:
        print("✅ fbgemm.asynchronous_complete_cumsum already available")

    # Step 8: Verify installation
    print("\n✅ Step 8: Verify Installation")

    # Add the package to sys.path explicitly
    if gen_rec_path not in sys.path:
        sys.path.insert(0, gen_rec_path)
        print(f"✅ Added {gen_rec_path} to sys.path")

    try:
        # Invalidate caches to ensure patched file is re-imported
        importlib.invalidate_caches()

        # Test Research HSTU imports
        from generative_recommenders.research.modeling.sequential.hstu import HSTU
        print("✅ Research HSTU imports successfully!")
        return True
    except ImportError as e:
        error_msg = str(e)
        print(f"❌ Import failed: {error_msg}")
        print("\n💡 Troubleshooting:")
        print("  1. Try restarting Colab runtime: Runtime → Restart runtime")
        print("  2. Delete /content/generative_recommenders and re-run setup")
        print(f"  3. Error details: {error_msg}")
        import traceback
        print(traceback.format_exc())
        return False

def quick_test():
    """Run a quick import test."""
    print("\n" + "=" * 70)
    print("Running Quick Import Test")
    print("=" * 70)

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")

        # Test Research HSTU imports
        from generative_recommenders.research.modeling.sequential.hstu import HSTU
        print("✅ Research HSTU")

        from generative_recommenders.research.modeling.sequential.embedding_modules import (
            LocalEmbeddingModule,
        )
        print("✅ LocalEmbeddingModule")

        from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
            LearnablePositionalEmbeddingInputFeaturesPreprocessor,
        )
        print("✅ LearnablePositionalEmbeddingInputFeaturesPreprocessor")

        from generative_recommenders.research.modeling.sequential.output_postprocessors import (
            L2NormEmbeddingPostprocessor,
        )
        print("✅ L2NormEmbeddingPostprocessor")

        from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (
            DotProductSimilarity,
        )
        print("✅ DotProductSimilarity")

        print("\n🎉 All imports successful! Ready to run integration test.")
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run setup
    success = setup_colab_environment()

    if success:
        # Run quick test
        if quick_test():
            print("\n" + "=" * 70)
            print("✅ Setup complete! You can now run:")
            print("   !python test_hstu_integration.py")
            print("=" * 70)
        else:
            print("\n⚠️  Setup completed but imports failed.")
            print("   Please restart Colab runtime and try again.")
    else:
        print("\n❌ Setup failed. Please check error messages above.")
