"""Utility functions for Uni-GCR."""

# Import utilities from parent-level utils.py module
# Note: This package (src/utils/) takes precedence over src/utils.py module,
# so we need to import from the parent level
import sys
from pathlib import Path

# Add parent src/ directory to be able to import utils.py
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import from utils.py at src/ level
from utils import (
    setup_distributed,
    is_main_process,
    set_seed,
    gather_tensors,
    compute_gr_metrics,
    compute_ctr_metrics
)

__all__ = [
    'setup_distributed',
    'is_main_process',
    'set_seed',
    'gather_tensors',
    'compute_gr_metrics',
    'compute_ctr_metrics'
]
