import numpy as np
from numba import cuda, uint16, uint32, uint64
import time

# SIMON 32/64 Constants
C_CONST = 0xFFFC
Z0_SEQ = np.array([
    1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,
    1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0
], dtype=np.uint16)

@cuda.jit(device=True)
def rotate_left(x, n):
    return ((x << n) & 0xFFFF) | (x >> (16 - n))

@cuda.jit(device=True)
def rotate_right(x, n):
    return (x >> n) | ((x << (16 - n)) & 0xFFFF)

@cuda.jit
def brute_force_kernel(target_ct, pt, known_key_part, bits_to_brute, z_gpu, offset, found_key_out):
    idx = cuda.grid(1)
    # The 'current_attempt' is the specific 40-bit combination for this thread
    current_attempt = uint64(idx) + uint64(offset)
    
    # Construct 64-bit key: [Known 24 bits] [Brute 40 bits]
    # We shift known part by 40 to make room for the brute force bits
    candidate_key = uint64((uint64(known_key_part) << bits_to_brute) | current_attempt)
    
    # Key Expansion (32 rounds)
    k = cuda.local.array(32, dtype=uint16)
    for i in range(4):
        k[i] = uint16((candidate_key >> (16 * i)) & 0xFFFF)
    for i in range(4, 32):
        tmp = rotate_right(k[i-1], 3) ^ k[i-3]
        tmp ^= rotate_right(tmp, 1)
        k[i] = uint16(k[i-4] ^ tmp ^ z_gpu[(i-4) % 62] ^ C_CONST)

    # Encryption Engine
    x = uint16((pt >> 16) & 0xFFFF)
    y = uint16(pt & 0xFFFF)
    for i in range(32):
        f_x = (rotate_left(x, 1) & rotate_left(x, 8)) ^ rotate_left(x, 2)
        new_x = uint16(y ^ f_x ^ k[i])
        y, x = x, new_x
    
    if (uint32(x) << 16 | uint32(y)) == target_ct:
        found_key_out[0] = candidate_key

def run_40bit_brute(target_ct, pt, known_key_24, known_len=24):
    bits_to_brute = 64 - known_len # 40 bits
    total_iterations = 1 << bits_to_brute
    
    # Batch size: 2^24 threads (~16.7M) per GPU launch to stay under TDR limits
    gpu_batch_size = 1 << 24 
    
    z_gpu = cuda.to_device(Z0_SEQ)
    found_key_out = cuda.to_device(np.zeros(1, dtype=np.uint64))
    
    threads_per_block = 256
    blocks_per_grid = gpu_batch_size // threads_per_block

    print(f"--- SIMON 32/64 Brute Force ---")
    print(f"Known Bits: {known_len} | Brute Bits: {bits_to_brute}")
    print(f"Total Search Space: 2^{bits_to_brute} ({total_iterations:,} keys)")
    
    start_time = time.time()
    
    # Main Loop: Iterate through the 40-bit space in batches
    for offset in range(0, total_iterations, gpu_batch_size):
        brute_force_kernel[blocks_per_grid, threads_per_block](
            uint32(target_ct), uint32(pt), uint64(known_key_24), 
            bits_to_brute, z_gpu, uint64(offset), found_key_out
        )
        
        # Check if key was found every few batches to save host-device sync time
        if (offset // gpu_batch_size) % 10 == 0:
            cuda.synchronize()
            res = found_key_out.copy_to_host()[0]
            if res != 0:
                print(f"\n[!] MATCH FOUND: {hex(res)}")
                print(f"Time Elapsed: {time.time() - start_time:.2f} seconds")
                return res
            
            # Progress calculation
            progress = (offset / total_iterations) * 100
            elapsed = time.time() - start_time
            print(f"Progress: {progress:.4f}% | Elapsed: {elapsed:.1f}s", end='\r')

    return None

# --- Configuration ---
TARGET_CT = 0xc69be9bb 
PT = 0x65656877
# Example: 24 bits known (0x191811)
KNOWN_24 = 0x191811 

result = run_40bit_brute(TARGET_CT, PT, KNOWN_24)
if not result:
    print("\nSearch complete. No key found.")
