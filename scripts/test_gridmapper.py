"""
Test script to verify GridMapper auto-detection works correctly
"""
import sys
sys.path.insert(0, '/Users/zhirong/GitHub/UniGCR_New')

from src.grid_utils import GridMapper

def test_gridmapper():
    print("="*70)
    print("TESTING GRIDMAPPER AUTO-DETECTION")
    print("="*70)

    # Test with auto-detection
    print("\n1. Testing auto-detection mode...")
    mapper = GridMapper(
        mapping_path='data/beauty/semantic_ids.json',
        auto_detect=True
    )

    print("\n2. Verifying structure...")
    assert mapper.num_layers == 4, f"Expected 4 layers, got {mapper.num_layers}"
    assert mapper.codebook_sizes == [256, 256, 256, 19], f"Wrong codebook sizes: {mapper.codebook_sizes}"
    assert mapper.total_vocab_size == 1 + 256 + 256 + 256 + 19, f"Wrong vocab size: {mapper.total_vocab_size}"

    print("✅ Structure verification passed!")
    print(f"   Layers: {mapper.num_layers}")
    print(f"   Codebook sizes: {mapper.codebook_sizes}")
    print(f"   Total vocab: {mapper.total_vocab_size}")

    print("\n3. Verifying layer ranges...")
    expected_ranges = [
        (1, 257),      # Layer 0: codes 0-255 mapped to vocab 1-256
        (257, 513),    # Layer 1: codes 0-255 mapped to vocab 257-512
        (513, 769),    # Layer 2: codes 0-255 mapped to vocab 513-768
        (769, 788)     # Layer 3: codes 0-18 mapped to vocab 769-787
    ]

    for i, (actual, expected) in enumerate(zip(mapper.layer_ranges, expected_ranges)):
        print(f"   Layer {i}: {actual} (expected {expected})")
        assert actual == expected, f"Layer {i} range mismatch!"

    print("✅ Layer ranges verified!")

    print("\n4. Testing semantic code conversion...")
    # Test with a few items
    test_items = [0, 1, 2, 100, 12100]

    for item_idx in test_items:
        codes = mapper.get_codes(item_idx)
        print(f"   Item {item_idx}: raw codes from json = {mapper.mapping.get(item_idx, 'N/A')}")
        print(f"              offset codes = {codes}")

        # Verify offset is applied correctly
        raw_codes = mapper.mapping.get(item_idx)
        if raw_codes:
            for layer_idx, (raw, offset) in enumerate(zip(raw_codes, codes)):
                expected_offset = mapper.layer_ranges[layer_idx][0] + raw
                assert offset == expected_offset, f"Wrong offset: {offset} != {expected_offset}"

    print("✅ Code conversion verified!")

    print("\n5. Testing reverse mapping...")
    # Test codes_to_item
    item_idx = 0
    codes = mapper.get_codes(item_idx)
    recovered_item = mapper.codes_to_item(codes)
    print(f"   Original item: {item_idx}")
    print(f"   Codes: {codes}")
    print(f"   Recovered item: {recovered_item}")
    assert recovered_item == item_idx, f"Reverse mapping failed: {recovered_item} != {item_idx}"

    print("✅ Reverse mapping verified!")

    print("\n6. Testing sequence flattening...")
    item_seq = [0, 1, 2]
    flat_seq = mapper.flatten_sequence(item_seq)
    print(f"   Item sequence: {item_seq}")
    print(f"   Flattened codes: {flat_seq}")
    print(f"   Length: {len(flat_seq)} (expected {len(item_seq) * mapper.num_layers})")
    assert len(flat_seq) == len(item_seq) * mapper.num_layers, "Wrong flattened length!"

    print("✅ Sequence flattening verified!")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)

    return mapper

if __name__ == "__main__":
    try:
        mapper = test_gridmapper()

        print("\n📊 SUMMARY:")
        print(f"   Semantic ID layers: {mapper.num_layers}")
        print(f"   Codebook sizes per layer: {mapper.codebook_sizes}")
        print(f"   Layer ranges: {mapper.layer_ranges}")
        print(f"   Total vocabulary size: {mapper.total_vocab_size}")
        print(f"   Number of items: {len(mapper.mapping)}")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
