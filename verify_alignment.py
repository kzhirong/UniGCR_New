"""
Verify alignment by loading directly from AmazonBeautyDataset.
No reimplementation — we call __getitem__ and inspect what the model actually sees.
"""
from src.config import UniGCRConfig
from src.data_amazon import AmazonBeautyDataset

config = UniGCRConfig()
config.data_path = 'data/train_sequences.json'

dataset = AmazonBeautyDataset(config, mode='train')
grid_mapper = dataset.grid_mapper
num_layers = config.sem_id_layers  # 4

print("=" * 70)
print("ALIGNMENT VERIFICATION  (using AmazonBeautyDataset directly)")
print("=" * 70)

for idx in range(3):
    sample = dataset[idx]

    sem_history = sample['sem_history']   # (152,)
    sem_target  = sample['sem_target']    # (152,)
    lengths     = sample['lengths'].item()  # tokens, e.g. 8 = 2 items
    target_idx  = sample['sem_target_eval'].item()

    num_hist_items = lengths // num_layers

    print(f"\n{'─'*70}")
    print(f"Sample {idx}  |  history items: {num_hist_items}  |  eval target item_id: {target_idx}")
    print(f"{'─'*70}")

    # --- raw token view (first 24 tokens) ---------------------------------
    print(f"\n[Token-level, first {min(24, lengths+8)} tokens]")
    print(f"  sem_history : {sem_history[:lengths + num_layers].tolist()}")
    print(f"  sem_target  : {sem_target[:lengths + num_layers].tolist()}")

    # --- item-level view --------------------------------------------------
    n_items = sem_history.size(0) // num_layers        # 38
    hist_items = sem_history.view(n_items, num_layers)
    tgt_items  = sem_target.view(n_items, num_layers)

    print(f"\n[Item-level view  (first {num_hist_items + 2} positions)]")
    print(f"  {'Pos':<4}  {'sem_history item':<28}  {'sem_target item':<28}  note")
    print(f"  {'---':<4}  {'----------------':<28}  {'---------------':<28}  ----")

    # get the actual target codes from grid_mapper for comparison
    gt_codes = grid_mapper.get_codes(target_idx)  # list of 4 ints (with offsets)

    for pos in range(min(num_hist_items + 2, n_items)):
        h = hist_items[pos].tolist()
        t = tgt_items[pos].tolist()

        # is this target item a proper 4-layer code? (all non-zero, correct ranges)
        ranges = grid_mapper.layer_ranges   # [(1,257),(257,513),(513,769),(769,788)]
        valid = all(ranges[i][0] <= t[i] < ranges[i][1] for i in range(num_layers))

        note = ""
        if t == gt_codes:
            note = "✓ CORRECT eval target"
        elif not valid and any(x != 0 for x in t):
            note = "✗ INVALID (garbage codes)"
        elif all(x == 0 for x in t):
            note = "  PAD"
        elif valid:
            note = "✓ valid item"

        print(f"  {pos:<4}  {str(h):<28}  {str(t):<28}  {note}")

    # --- key question: does sem_target at position (num_hist_items-1) equal the eval target?
    last_hist_pos = num_hist_items - 1
    target_in_sem = tgt_items[last_hist_pos].tolist()
    print(f"\n[Key check]")
    print(f"  sem_target at last history position ({last_hist_pos}) : {target_in_sem}")
    print(f"  ground-truth eval target codes                       : {gt_codes}")
    if target_in_sem == gt_codes:
        print(f"  ✓ MATCH — model will learn the right thing at eval position")
    else:
        print(f"  ✗ MISMATCH — model still trains on wrong target at eval position")

print(f"\n{'='*70}")
print("DONE")
print("="*70)
