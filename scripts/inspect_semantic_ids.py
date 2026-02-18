"""
Script to inspect your .pt semantic ID file and convert it to required JSON format
"""
import torch
import json
import sys

def inspect_pt_file(pt_path):
    """
    Loads and inspects the .pt file to understand its structure
    """
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path)

    print(f"\nType: {type(data)}")

    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:10]}...")  # Show first 10 keys
        first_key = list(data.keys())[0]
        print(f"Sample entry: {first_key} -> {data[first_key]}")
        print(f"Total items: {len(data)}")
    elif isinstance(data, torch.Tensor):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Sample values:\n{data[:5]}")
    else:
        print(f"Data structure: {data}")

    return data

def convert_to_json(data, output_path):
    """
    Converts the .pt data to the required JSON format:
    { "item_id": [code1, code2, code3], ... }
    """
    if isinstance(data, dict):
        # Check if values are already in list format or tensors
        converted = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                converted[str(k)] = v.tolist()
            elif isinstance(v, list):
                converted[str(k)] = v
            else:
                converted[str(k)] = [v]  # Wrap single values

        with open(output_path, 'w') as f:
            json.dump(converted, f, indent=2)

        print(f"\n✅ Converted {len(converted)} items to {output_path}")
        print(f"Sample: {list(converted.items())[0]}")
        return True
    else:
        print("❌ Data format not recognized. Expected dict mapping item_id -> codes")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_semantic_ids.py <path_to_pt_file> [output_json_path]")
        sys.exit(1)

    pt_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/beauty/semantic_ids.json"

    data = inspect_pt_file(pt_path)

    print("\n" + "="*50)
    convert = input("Convert to JSON? (y/n): ")

    if convert.lower() == 'y':
        convert_to_json(data, output_path)
