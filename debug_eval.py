"""
Debug evaluation to understand why val_loss improves but Hit@10 stays at 0
"""
import torch
from src.config import UniGCRConfig
from src.model import UniGCRModel
from src.data import get_dataloaders
from src.grid_utils import GridMapper
from tqdm import tqdm

def debug_evaluation():
    # Load config and model
    config = UniGCRConfig()
    config.device = "cuda"

    # Load grid mapper
    grid_mapper = GridMapper(config.grid_mapping_path)
    config.sem_total_vocab = grid_mapper.total_vocab_size

    # Load model
    model = UniGCRModel(config).to(config.device)
    checkpoint = torch.load("checkpoints/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load validation data
    _, val_loader, _ = get_dataloaders(config, grid_mapper)

    print("=" * 80)
    print("DEBUGGING EVALUATION PIPELINE")
    print("=" * 80)

    # Analyze first batch
    batch = next(iter(val_loader))
    batch = {k: v.to(config.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        # 1. Generate candidates (beam search + nearest neighbor)
        candidates = model.generate_gr_candidates(batch, k=10, grid_mapper=grid_mapper)

        # 2. Get ground truth
        target_item_ids = batch.get('sem_target_eval', None)

        print(f"\n📊 Batch Statistics:")
        print(f"   Batch size: {candidates.size(0)}")
        print(f"   Candidates per user: {candidates.size(1)}")

        # 3. Check beam diversity
        unique_counts = []
        for i in range(min(5, candidates.size(0))):  # First 5 users
            unique_items = torch.unique(candidates[i])
            unique_counts.append(len(unique_items))

            print(f"\n👤 User {i}:")
            print(f"   Ground truth item: {target_item_ids[i].item()}")
            print(f"   Top-10 predictions: {candidates[i].tolist()}")
            print(f"   Unique items in top-10: {len(unique_items)}/10")
            print(f"   Is target in top-10? {(target_item_ids[i] in candidates[i]).item()}")

        avg_unique = sum(unique_counts) / len(unique_counts)
        print(f"\n📈 Average unique items per user: {avg_unique:.2f}/10")

        # 4. Check what codes are being predicted
        print(f"\n🔍 Raw Beam Search Codes (before nearest neighbor):")
        u, _, _ = model.forward(batch)
        beam_codes = model._beam_search_hard_negatives(
            batch, u[:5], beam_width=10, grid_mapper=grid_mapper
        )

        for i in range(min(3, beam_codes.size(0))):
            print(f"\n   User {i} predicted codes:")
            for j in range(5):  # First 5 beams
                code = beam_codes[i, j].tolist()

                # Apply offset and map to item
                offsets = torch.tensor([1, 257, 513, 769], device=beam_codes.device)
                code_offset = beam_codes[i, j] + offsets
                item_id = grid_mapper.codes_to_item_nearest_batch(code_offset.unsqueeze(0))[0].item()

                print(f"      Beam {j}: {code} → Item {item_id}")

        # 5. Check ground truth codes
        print(f"\n🎯 Ground Truth Codes:")
        sem_target = batch['sem_target']
        for i in range(min(3, sem_target.size(0))):
            # Get ground truth codes for first item (remove offset)
            gt_codes_offset = sem_target[i, 0].tolist()
            gt_codes_raw = [
                gt_codes_offset[0] - 1,
                gt_codes_offset[1] - 257,
                gt_codes_offset[2] - 513,
                gt_codes_offset[3] - 769
            ]
            gt_item = target_item_ids[i].item()
            print(f"   User {i}: GT Item {gt_item} has codes {gt_codes_raw}")

        # 6. Overall Hit@10 for this batch
        hits = 0
        for i in range(target_item_ids.size(0)):
            if target_item_ids[i] in candidates[i]:
                hits += 1
        hit_rate = hits / target_item_ids.size(0)
        print(f"\n✅ Hit@10 for this batch: {hit_rate:.4f} ({hits}/{target_item_ids.size(0)})")

        # 7. Check if it's a systematic bias
        print(f"\n🔍 Checking for systematic bias:")
        all_predicted_items = candidates.view(-1).cpu().numpy()
        print(f"   Total predictions: {len(all_predicted_items)}")
        print(f"   Unique items predicted: {len(set(all_predicted_items))}")
        print(f"   Most common predictions: {sorted(set(all_predicted_items), key=lambda x: list(all_predicted_items).count(x), reverse=True)[:10]}")

if __name__ == "__main__":
    debug_evaluation()
