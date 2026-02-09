"""
Comprehensive diagnostic to understand why Hit@10 = 0

This will check:
1. What codes are being predicted vs ground truth
2. Are predicted codes close to ground truth? (per-layer accuracy)
3. Beam diversity - are all beams collapsing to same codes?
4. After nearest neighbor mapping - are all beams mapping to same item?
5. Are predicted codes valid (exist in training data)?
6. Distribution analysis - is model biased toward certain codes?
"""
import torch
import json
from collections import Counter
from src.config import UniGCRConfig
from src.model import UniGCRModel
from src.data import get_dataloaders
from src.grid_utils import GridMapper

def diagnose():
    print("=" * 80)
    print("DIAGNOSTIC: Understanding Why Hit@10 = 0")
    print("=" * 80)

    # Load config
    config = UniGCRConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load grid mapper
    grid_mapper = GridMapper(config.grid_mapping_path)
    config.sem_total_vocab = grid_mapper.total_vocab_size

    # Load model
    print("\n📦 Loading model from checkpoint...")
    model = UniGCRModel(config).to(config.device)
    checkpoint = torch.load("checkpoints/best_model.pt", map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Load validation data
    _, val_loader, _ = get_dataloaders(config, grid_mapper)

    # Get first batch for analysis
    batch = next(iter(val_loader))
    batch = {k: v.to(config.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    B = batch['sem_target'].size(0)
    print(f"\n📊 Analyzing batch of {B} users")

    with torch.no_grad():
        # ============================================================
        # 1. Get ground truth codes
        # ============================================================
        sem_target = batch['sem_target']  # (B, num_items, 4) with offsets
        target_item_ids = batch['sem_target_eval']  # (B,)

        # Ground truth codes for first item (what model should predict)
        gt_codes_offset = sem_target[:, 0, :]  # (B, 4)
        gt_codes_raw = gt_codes_offset - torch.tensor([1, 257, 513, 769], device=config.device)

        print(f"\n🎯 Ground Truth Examples (first 5 users):")
        for i in range(min(5, B)):
            gt_codes = gt_codes_raw[i].tolist()
            gt_item = target_item_ids[i].item()
            print(f"   User {i}: Item={gt_item:5d}, Codes={gt_codes}")

        # ============================================================
        # 2. Get model predictions (autoregressive, no teacher forcing)
        # ============================================================
        u, _, _ = model.forward(batch)

        # Predict codes autoregressively (evaluation mode - greedy)
        print(f"\n🤖 Model Predictions (autoregressive, greedy):")
        logits_list, sampled_codes_list = model.predict_codes_autoregressive(
            u, target_codes=None, training=False
        )

        # sampled_codes_list: list of 4 tensors, each (B,)
        pred_codes_raw = torch.stack(sampled_codes_list, dim=1)  # (B, 4)

        for i in range(min(5, B)):
            pred_codes = pred_codes_raw[i].tolist()
            gt_codes = gt_codes_raw[i].tolist()
            matches = [p == g for p, g in zip(pred_codes, gt_codes)]
            print(f"   User {i}: Pred={pred_codes}, GT={gt_codes}, Match={matches}")

        # ============================================================
        # 3. Per-layer prediction accuracy
        # ============================================================
        print(f"\n📈 Per-Layer Prediction Accuracy:")
        layer_names = ["L0", "L1", "L2", "Dedup"]
        for layer_idx in range(4):
            correct = (pred_codes_raw[:, layer_idx] == gt_codes_raw[:, layer_idx]).sum().item()
            accuracy = correct / B
            print(f"   {layer_names[layer_idx]}: {correct}/{B} correct ({accuracy*100:.1f}%)")

        # Full sequence accuracy (all 4 layers correct)
        full_match = (pred_codes_raw == gt_codes_raw).all(dim=1).sum().item()
        print(f"   Full Sequence: {full_match}/{B} correct ({full_match/B*100:.1f}%)")

        # ============================================================
        # 4. Beam search analysis
        # ============================================================
        print(f"\n🔍 Beam Search Analysis (k=10):")
        beam_codes = model._beam_search_hard_negatives(
            batch, u, beam_width=10, grid_mapper=grid_mapper
        )  # (B, 10, 4)

        # Check diversity of beam codes
        unique_per_user = []
        for i in range(min(5, B)):
            beam_codes_i = beam_codes[i]  # (10, 4)
            # Convert to tuples for uniqueness check
            beam_tuples = [tuple(beam_codes_i[j].tolist()) for j in range(10)]
            unique_codes = len(set(beam_tuples))
            unique_per_user.append(unique_codes)

            gt_codes = gt_codes_raw[i].tolist()
            gt_tuple = tuple(gt_codes)

            print(f"\n   User {i}:")
            print(f"     Ground truth: {gt_codes}")
            print(f"     Unique codes in top-10 beams: {unique_codes}/10")
            print(f"     Is GT in beams? {gt_tuple in beam_tuples}")
            print(f"     Top 3 beam codes:")
            for j in range(3):
                print(f"       Beam {j}: {beam_codes_i[j].tolist()}")

        avg_unique_codes = sum(unique_per_user) / len(unique_per_user)
        print(f"\n   ⚠️  Average unique codes per user: {avg_unique_codes:.1f}/10")
        if avg_unique_codes < 5:
            print(f"   🚨 BEAM COLLAPSE! Beams are not diverse!")

        # ============================================================
        # 5. After nearest neighbor mapping
        # ============================================================
        print(f"\n🗺️  After Nearest Neighbor Mapping:")

        # Map beam codes to items
        offsets = torch.tensor([1, 257, 513, 769], device=config.device)
        beam_codes_flat = beam_codes.view(-1, 4)  # (B*10, 4)
        beam_codes_offset = beam_codes_flat + offsets
        predicted_items = grid_mapper.codes_to_item_nearest_batch(beam_codes_offset)
        predicted_items = predicted_items.view(B, 10)  # (B, 10)

        unique_items_per_user = []
        for i in range(min(5, B)):
            predicted_items_i = predicted_items[i]  # (10,)
            unique_items = len(torch.unique(predicted_items_i))
            unique_items_per_user.append(unique_items)

            gt_item = target_item_ids[i].item()
            is_hit = (gt_item in predicted_items_i).item()

            print(f"\n   User {i}:")
            print(f"     Ground truth item: {gt_item}")
            print(f"     Predicted items: {predicted_items_i.tolist()}")
            print(f"     Unique items: {unique_items}/10")
            print(f"     Hit@10: {is_hit}")

        avg_unique_items = sum(unique_items_per_user) / len(unique_items_per_user)
        print(f"\n   ⚠️  Average unique items per user: {avg_unique_items:.1f}/10")
        if avg_unique_items < 5:
            print(f"   🚨 ITEM COLLAPSE! Many beams map to same item!")

        # ============================================================
        # 6. Overall Hit@10 for this batch
        # ============================================================
        hits = 0
        for i in range(B):
            if target_item_ids[i] in predicted_items[i]:
                hits += 1
        hit_rate = hits / B

        print(f"\n✅ Overall Hit@10 for batch: {hit_rate:.4f} ({hits}/{B})")

        # ============================================================
        # 7. Check prediction distribution bias
        # ============================================================
        print(f"\n📊 Prediction Distribution Analysis:")

        # L0 distribution
        l0_pred_counts = Counter(pred_codes_raw[:, 0].cpu().tolist())
        l0_gt_counts = Counter(gt_codes_raw[:, 0].cpu().tolist())

        print(f"\n   L0 Predictions:")
        print(f"     Unique values predicted: {len(l0_pred_counts)}/256")
        print(f"     Top 5 most common: {l0_pred_counts.most_common(5)}")
        print(f"   L0 Ground Truth:")
        print(f"     Unique values: {len(l0_gt_counts)}/256")
        print(f"     Top 5 most common: {l0_gt_counts.most_common(5)}")

        if len(l0_pred_counts) < 50:
            print(f"   🚨 MODEL COLLAPSE! Only predicting {len(l0_pred_counts)} unique L0 codes!")

        # ============================================================
        # 8. Check if predicted codes are valid
        # ============================================================
        print(f"\n🔍 Validity Check:")

        # Load all valid code combinations
        with open(config.grid_mapping_path, 'r') as f:
            semantic_ids = json.load(f)

        valid_codes = set()
        for item_data in semantic_ids['item_to_codes'].values():
            codes = tuple(item_data['codes'])
            valid_codes.add(codes)

        print(f"   Total valid code combinations: {len(valid_codes)}")

        # Check if predictions are valid
        invalid_count = 0
        for i in range(B):
            pred_tuple = tuple(pred_codes_raw[i].tolist())
            if pred_tuple not in valid_codes:
                invalid_count += 1

        print(f"   Invalid predictions: {invalid_count}/{B} ({invalid_count/B*100:.1f}%)")
        if invalid_count > B * 0.5:
            print(f"   🚨 MAJOR ISSUE! Model predicting codes that don't exist!")

        # ============================================================
        # 9. Distance analysis
        # ============================================================
        print(f"\n📏 Distance from Predictions to Ground Truth:")

        # Hamming distance per layer
        for layer_idx, layer_name in enumerate(layer_names):
            distances = torch.abs(pred_codes_raw[:, layer_idx] - gt_codes_raw[:, layer_idx]).float()
            avg_dist = distances.mean().item()
            print(f"   {layer_name}: avg distance = {avg_dist:.2f}")

        print("\n" + "=" * 80)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    diagnose()
