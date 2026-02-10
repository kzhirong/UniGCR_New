"""
Trace data flow to understand training/evaluation mismatch
"""

# Simulate data preparation
def simulate_data_prep():
    # Example: User has 2 history items, each with 4 codes
    # History: item1=[10,20,30,0], item2=[40,50,60,1]
    # Target: item3=[70,80,90,2]

    history_codes = [10, 20, 30, 0, 40, 50, 60, 1]  # 8 tokens
    target_codes = [70, 80, 90, 2]  # 4 tokens

    # full_seq = history + target
    full_seq = history_codes + target_codes  # 12 tokens
    print(f"full_seq (12 tokens): {full_seq}")

    # Pad to max_len = 20
    max_len = 20
    padded_left = [0] * (max_len - len(full_seq)) + full_seq  # OLD: left-padding
    padded_right = full_seq + [0] * (max_len - len(full_seq))  # NEW: right-padding

    print(f"\nOLD (left-padded):  {padded_left}")
    print(f"  Positions 0-7: {padded_left[0:8]} (padding)")
    print(f"  Positions 8-19: {padded_left[8:20]} (actual data)")

    print(f"\nNEW (right-padded): {padded_right}")
    print(f"  Positions 0-11: {padded_right[0:12]} (actual data)")
    print(f"  Positions 12-19: {padded_right[12:20]} (padding)")

    padded = padded_right  # Use right-padding

    # BEFORE FIX: actual_length = len(full_seq) - 1 = 11 (WRONG - includes partial target!)
    # AFTER FIX: actual_length = len(history_codes) = 8 (CORRECT - history only!)
    actual_length_old = len(full_seq) - 1
    actual_length_new = len(history_codes)
    print(f"\nOLD actual_length (buggy): {actual_length_old} tokens")
    print(f"NEW actual_length (fixed): {actual_length_new} tokens (history only)")

    actual_length = actual_length_new  # Use new fixed version

    # Split into input and target
    sem_history = padded[:-1]  # [:-1] removes last token
    sem_target = padded[1:]     # [1:] shifts by 1

    print(f"\nsem_history (19 tokens): {sem_history}")
    print(f"  Last valid position (lengths-1): {actual_length - 1}")
    print(f"  sem_history[{actual_length - 1}]: {sem_history[actual_length - 1]}")

    print(f"\nsem_target (19 tokens): {sem_target}")
    print(f"  Position 0: {sem_target[0]} (padding)")
    print(f"  Position 7: {sem_target[7]} (padding)")
    print(f"  Position 8: {sem_target[8]} (first real token)")
    print(f"  Position 18: {sem_target[18]} (last token = target_L3)")

    # During training, model predicts at each position
    print(f"\nTraining predictions:")
    for pos in range(len(sem_history)):
        input_token = sem_history[pos]
        target_token = sem_target[pos]
        is_padding = pos < (max_len - len(full_seq))
        print(f"  pos={pos}: input={input_token:3d} -> predict={target_token:3d} {'(PADDING, ignored with -100)' if is_padding else ''}")

    # During evaluation, model uses last valid position
    # Model converts token-level length to item-level: lengths_items = actual_length // 4
    lengths_items_old = actual_length_old // 4
    lengths_items_new = actual_length_new // 4

    last_position_old = lengths_items_old - 1
    last_position_new = lengths_items_new - 1

    print(f"\nEvaluation:")
    print(f"\n  OLD (buggy):")
    print(f"    Token-level length: {actual_length_old}")
    print(f"    Item-level length: {lengths_items_old} (= {actual_length_old} // 4)")
    print(f"    last_position: {last_position_old} (= {lengths_items_old} - 1)")
    print(f"    sem_history[{last_position_old}]: {sem_history[last_position_old]} (WRONG - middle of sequence!)")

    print(f"\n  NEW (fixed):")
    print(f"    Token-level length: {actual_length_new}")
    print(f"    Item-level length: {lengths_items_new} (= {actual_length_new} // 4)")
    print(f"    last_position (item index): {last_position_new} (= {lengths_items_new} - 1)")

    # Show which ITEM gets extracted
    item_start = last_position_new * 4
    item_end = item_start + 4
    extracted_item = sem_history[item_start:item_end]
    print(f"    Extracted item (tokens {item_start}-{item_end-1}): {extracted_item}")

    # Compare with ground truth
    expected_item = [40, 50, 60, 1]  # item2 (last history item)
    if extracted_item == expected_item:
        print(f"    ✅ CORRECT! This is item2 (last history item)")
    else:
        print(f"    ❌ WRONG! Expected item2: {expected_item}")

    print(f"\n  🎉 Model now correctly extracts from the LAST HISTORY item to predict next item!")

if __name__ == "__main__":
    simulate_data_prep()
