# save as: weights_bin_dump.py
import argparse, numpy as np, sys

def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    # raw_u16: little-endian BF16 words
    return ((raw_u16.astype(np.uint32) << 16)).view('<f4')

p = argparse.ArgumentParser(description="Dump BF16 weights.bin as readable floats")
p.add_argument("path", help="path to weights.bin")
p.add_argument("--elem-offset", type=int, default=0, help="starting element (BF16) offset")
p.add_argument("--byte-offset", type=int, default=None, help="starting byte offset (overrides elem-offset)")
p.add_argument("--count", type=int, default=64, help="how many BF16 elements to read (-1 = all)")
p.add_argument("--shape", default=None, help="reshape as 'rows,cols' (optional)")
p.add_argument("--slice", default=None, help="print sub-block like 'r0:r1,c0:c1' (optional)")
p.add_argument("--save", default=None, help="save the (possibly reshaped) array as .npy")
args = p.parse_args()

# Open and read
if args.count == -1 and args.byte_offset is None and args.elem_offset == 0:
    # read entire file
    raw = np.fromfile(args.path, dtype='<u2')  # BF16 words
else:
    # read only a chunk
    start_bytes = args.byte_offset if args.byte_offset is not None else args.elem_offset * 2
    if args.count == -1:
        # read to end
        with open(args.path, "rb") as f:
            f.seek(start_bytes)
            buf = f.read()
        raw = np.frombuffer(buf, dtype='<u2')
    else:
        with open(args.path, "rb") as f:
            f.seek(start_bytes)
            buf = f.read(args.count * 2)
        if len(buf) < (args.count * 2):
            print(f"Warning: requested {args.count} elements, got only {len(buf)//2}", file=sys.stderr)
        raw = np.frombuffer(buf, dtype='<u2')

f32 = bf16_to_f32(raw)

# Optional reshape
arr = f32
if args.shape:
    r, c = map(int, args.shape.split(","))
    total = r * c
    if total > arr.size:
        print(f"Error: requested shape {r}x{c} needs {total} elements, but only {arr.size} loaded.", file=sys.stderr)
        sys.exit(1)
    arr = arr[:total].reshape(r, c)

# Optional slicing
if args.slice:
    rs, cs = args.slice.split(",")
    def parse(sl):
        a, b = (sl.split(":")+["",""])[:2]
        return slice(int(a) if a else None, int(b) if b else None)
    arr = arr[parse(rs), parse(cs)]

# Output
np.set_printoptions(precision=6, suppress=True, linewidth=180)
print(arr)

if args.save:
    np.save(args.save, arr)
    print("Saved:", args.save)

