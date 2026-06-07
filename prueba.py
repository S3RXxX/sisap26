import h5py

file = "/home/ivan/.cache/huggingface/hub/datasets--SISAP-Challenges--SISAP2026/snapshots/ebc9e713a296f0aff6b5109c8c1961b3b75caadf/wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5"
with h5py.File(file, "r") as f:
    print(list(f.keys()))
    def print_structure(name, obj):
        indent = "    " * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f"{indent}📁 {name}  [GROUP]")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}📊 {name}  [DATASET]  shape={obj.shape}  dtype={obj.dtype}")
    
    f.visititems(print_structure)