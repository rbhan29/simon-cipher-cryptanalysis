import numpy as np
from numba import cuda, uint16, uint32, uint64
import time
import os

# SIMON 32/64 Constants
C_CONST = 0xFFFC
Z0_SEQ = np.array([1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,
                   1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0], dtype=np.uint16)

@cuda.jit(device=True)
def rotate_left(x, n):
    return ((x << n) & 0xFFFF) | (x >> (16 - n))

@cuda.jit(device=True)
def rotate_right(x, n):
    return (x >> n) | ((x << (16 - n)) & 0xFFFF)

@cuda.jit
def brute_force_kernel(target_ct, pt, known_key_16, bits_to_brute, z_gpu, offset, found_key_out):
    idx = cuda.grid(1)
    current_attempt = uint64(idx) + uint64(offset)
    
    # Construct 64-bit key: [Known 16 bits] [Brute 48 bits]
    candidate_key = uint64((uint64(known_key_16) << bits_to_brute) | current_attempt)
    
    # Key Expansion
    k = cuda.local.array(32, dtype=uint16)
    for i in range(4):
        k[i] = uint16((candidate_key >> (16 * i)) & 0xFFFF)
    for i in range(4, 32):
        tmp = rotate_right(k[i-1], 3) ^ k[i-3]
        tmp ^= rotate_right(tmp, 1)
        k[i] = uint16(k[i-4] ^ tmp ^ z_gpu[(i-4) % 62] ^ C_CONST)

    # Encryption
    x = uint16((pt >> 16) & 0xFFFF)
    y = uint16(pt & 0xFFFF)
    for i in range(32):
        f_x = (rotate_left(x, 1) & rotate_left(x, 8)) ^ rotate_left(x, 2)
        new_x = uint16(y ^ f_x ^ k[i])
        y, x = x, new_x
    
    if (uint32(x) << 16 | uint32(y)) == target_ct:
        found_key_out[0] = candidate_key

def run_48bit_brute(target_ct, pt, known_16):
    bits_to_brute = 48
    total_iterations = 1 << bits_to_brute
    gpu_batch_size = 1 << 26 # Larger batch for better GPU utilization (~67M keys)
    
    # Checkpoint logic
    checkpoint_file = "brute_checkpoint.txt"
    start_offset = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_offset = int(f.read().strip())
        print(f"Resuming from checkpoint: {start_offset:,}")

    z_gpu = cuda.to_device(Z0_SEQ)
    found_key_out = cuda.to_device(np.zeros(1, dtype=np.uint64))
    
    threads_per_block = 256
    blocks_per_grid = gpu_batch_size // threads_per_block

    print(f"Brute forcing {bits_to_brute} bits on RTX 5050...")
    start_time = time.time()
    
    try:
        for offset in range(start_offset, total_iterations, gpu_batch_size):
            brute_force_kernel[blocks_per_grid, threads_per_block](
                uint32(target_ct), uint32(pt), uint64(known_16), 
                bits_to_brute, z_gpu, uint64(offset), found_key_out
            )
            
            # Synchronize and check every batch (at 48-bit, every check matters)
            cuda.synchronize()
            res = found_key_out.copy_to_host()
            if res[0] != 0:
                print(f"\n[!] KEY FOUND: {hex(res[0])}")
                return res[0]
            
            # Save progress every 10 batches (~670M keys)
            if (offset // gpu_batch_size) % 10 == 0:
                with open(checkpoint_file, "w") as f:
                    f.write(str(offset))
                
                # Stats
                progress = (offset / total_iterations) * 100
                keys_per_sec = (offset - start_offset + 1) / (time.time() - start_time)
                print(f"Progress: {progress:.6f}% | Speed: {keys_per_sec/1e6:.2f}M keys/s", end='\r')

    except KeyboardInterrupt:
        print("\nPaused. Checkpoint saved.")
    
    return None

# --- Configuration ---
TARGET_CT = 0xc69be9bb
PT = 0x65656877
KNOWN_16 = 0x1918  # Prefix of 0x1918111009080100

result = run_48bit_brute(TARGET_CT, PT, KNOWN_16)
