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
    current_attempt = uint64(idx) + uint64(offset)
    
    # Construct 64-bit key: [Known 32 bits] [Brute 32 bits]
    candidate_key = uint64((known_key_part << bits_to_brute) | current_attempt)
    
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

def run_large_brute(target_ct, pt, known_key, known_len):
    bits_to_brute = 64 - known_len
    total_iterations = 1 << bits_to_brute
    
    # Batch size (2^24 is ~16 million threads per launch to avoid timeouts)
    batch_size = 1 << 24 
    z_gpu = cuda.to_device(Z0_SEQ)
    found_key_out = cuda.to_device(np.zeros(1, dtype=np.uint64))
    
    threads_per_block = 256
    blocks_per_grid = batch_size // threads_per_block

    print(f"Targeting {total_iterations} combinations in batches of {batch_size}...")
    
    start_time = time.time()
    for offset in range(0, total_iterations, batch_size):
        brute_force_kernel[blocks_per_grid, threads_per_block](
            uint32(target_ct), uint32(pt), uint64(known_key), 
            bits_to_brute, z_gpu, uint64(offset), found_key_out
        )
        cuda.synchronize()
        
        # Check if found
        res = found_key_out.copy_to_host()
        if res[0] != 0:
            print(f"\nMatch found in {(time.time() - start_time):.2f}s!")
            return res[0]
            
        print(f"Progress: {(offset + batch_size) / total_iterations * 100:.1f}%", end='\r')

    return None

# --- Configuration for 32-bit Brute Force ---
# Target CT: 0x2287299a (Random example, replace with your actual target)
# Known Part (High 32 bits): 0x19181110
KNOWN_PART = 0x19181110 
KNOWN_LEN = 32
TARGET_CT = 0xc69be9bb 
PT = 0x65656877

result = run_large_brute(TARGET_CT, PT, KNOWN_PART, KNOWN_LEN)
print(f"Final Result: {hex(result) if result else 'Not Found'}")
