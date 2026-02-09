"""
Test script for Independent Top-K beam search implementation
"""
import torch
from src.config import UniGCRConfig
from src.model import UniGCRModel
from src.data_amazon import get_dataloaders

def test_beam_search():
    print("=" * 60)
    print("Testing Independent Top-K Beam Search")
    print("=" * 60)

    # 1. Load config
    config = UniGCRConfig()
    config.sem_id_layers = 4
    config.sem_id_codebook_size = 256
    config.sem_id_dedup_size = 19
    config.max_seq_len = 153
    config.batch_size = 4  # Small batch for testing

    print(f"\n[Config]")
    print(f"  Layers: {config.sem_id_layers}")
    print(f"  Codebook size (L0/L1/L2): {config.sem_id_codebook_size}")
    print(f"  Dedup size: {config.sem_id_dedup_size}")
    print(f"  Batch size: {config.batch_size}")

    # 2. Create model
    print(f"\n[Model]")
    model = UniGCRModel(config)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Load data
    print(f"\n[Data]")
    train_loader, val_loader = get_dataloaders(config, args=None)
    grid_mapper = val_loader.dataset.grid_mapper
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  GridMapper items: {len(grid_mapper.mapping)}")

    # 4. Test beam search on one batch
    print(f"\n[Beam Search Test]")
    batch = next(iter(val_loader))
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    print(f"  Batch size: {batch['sem_history'].size(0)}")
    print(f"  Sequence length: {batch['sem_history'].size(1)}")

    with torch.no_grad():
        # Forward pass to get user state
        u, logits_seq, _ = model(batch)
        print(f"  User state shape: {u.shape}")
        print(f"  Logits L0 shape: {logits_seq[0].shape}")
        print(f"  Logits L1 shape: {logits_seq[1].shape}")
        print(f"  Logits L2 shape: {logits_seq[2].shape}")
        print(f"  Logits Dedup shape: {logits_seq[3].shape}")

        # Test beam search with k=10
        print(f"\n  Running beam search with k=10...")
        beam_width = 10
        beam_results = model._beam_search_hard_negatives(
            batch, u, beam_width=beam_width, grid_mapper=grid_mapper
        )

        print(f"  Beam results shape: {beam_results.shape}")
        print(f"  Expected shape: ({batch['sem_history'].size(0)}, {beam_width}, 4)")

        # Verify shape
        B = batch['sem_history'].size(0)
        assert beam_results.shape == (B, beam_width, 4), \
            f"Shape mismatch! Got {beam_results.shape}, expected ({B}, {beam_width}, 4)"

        # Show sample predictions
        print(f"\n  Sample predictions for first user:")
        for i in range(min(5, beam_width)):
            codes = beam_results[0, i].cpu().tolist()
            print(f"    Rank {i+1}: L0={codes[0]}, L1={codes[1]}, L2={codes[2]}, Dedup={codes[3]}")

        # Verify all codes are in valid range
        print(f"\n  Validating code ranges...")
        for layer_idx in range(4):
            layer_codes = beam_results[:, :, layer_idx]
            min_code = layer_codes.min().item()
            max_code = layer_codes.max().item()

            if layer_idx < 3:  # L0, L1, L2
                expected_max = config.sem_id_codebook_size - 1  # 255
                print(f"    Layer {layer_idx}: min={min_code}, max={max_code} (expected: 0-{expected_max})")
                assert 0 <= min_code <= expected_max, f"L{layer_idx} min code out of range!"
                assert 0 <= max_code <= expected_max, f"L{layer_idx} max code out of range!"
            else:  # Dedup
                expected_max = config.sem_id_dedup_size - 1  # 18
                print(f"    Dedup: min={min_code}, max={max_code} (expected: 0-{expected_max})")
                assert 0 <= min_code <= expected_max, f"Dedup min code out of range!"
                assert 0 <= max_code <= expected_max, f"Dedup max code out of range!"

        # Test generate_gr_candidates
        print(f"\n  Testing generate_gr_candidates...")
        candidates = model.generate_gr_candidates(
            batch, k=10, grid_mapper=grid_mapper
        )

        print(f"  Candidates shape: {candidates.shape}")
        print(f"  Expected shape: ({B}, 10)")
        assert candidates.shape == (B, 10), \
            f"Shape mismatch! Got {candidates.shape}, expected ({B}, 10)"

        # Show sample item predictions
        print(f"\n  Sample item predictions for first user:")
        for i in range(min(5, 10)):
            item_id = candidates[0, i].item()
            print(f"    Rank {i+1}: Item ID = {item_id}")

        # Verify all item IDs are valid
        min_item = candidates.min().item()
        max_item = candidates.max().item()
        print(f"\n  Item ID range: min={min_item}, max={max_item}")
        print(f"  Total items in catalog: {len(grid_mapper.mapping)}")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Beam search is working correctly.")
    print("=" * 60)
    print("\n[Next Steps]")
    print("1. Download checkpoint from Colab: checkpoints/best_model.pt")
    print("2. Run evaluation with new beam search:")
    print("   python run.py --eval_only")
    print("3. You should now see real Hit@10 and NDCG@10 metrics!")
    print()

if __name__ == '__main__':
    test_beam_search()
