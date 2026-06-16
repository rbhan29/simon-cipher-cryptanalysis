import numpy as np
from numba import cuda, uint16, uint32, uint64

# SIMON 32/64 Constants
C_CONST = 0xFFFC
# The Z0 sequence for SIMON 32/64
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
def brute_force_kernel(target_ct, pt, known_key_part, bits_to_brute, z_gpu, found_key_out):
    idx = cuda.grid(1)
    max_iter = 1 << bits_to_brute
    
    if idx >= max_iter:
        return

    # 1. Construct the full 64-bit key candidate
    # known_key_part shifted left, idx filled in the 'empty' bits
    candidate_key = uint64((known_key_part << bits_to_brute) | idx)
    
    # 2. Key Expansion (Using a Local Array instead of a Python List)
    # n-bit words (n=16), m=4 keywords
    k = cuda.local.array(32, dtype=uint16)
    
    for i in range(4):
        k[i] = uint16((candidate_key >> (16 * i)) & 0xFFFF)
        
    for i in range(4, 32):
        tmp = rotate_right(k[i-1], 3)
        tmp ^= k[i-3]
        tmp ^= rotate_right(tmp, 1)
        k[i] = uint16(k[i-4] ^ tmp ^ z_gpu[(i-4) % 62] ^ C_CONST)

    # 3. Encryption Engine
    x = uint16((pt >> 16) & 0xFFFF)
    y = uint16(pt & 0xFFFF)
    
    for i in range(32):
        f_x = (rotate_left(x, 1) & rotate_left(x, 8)) ^ rotate_left(x, 2)
        new_x = uint16(y ^ f_x ^ k[i])
        y = x
        x = new_x
    
    result_ct = (uint32(x) << 16) | uint32(y)
    
    # 4. Success Check
    if result_ct == target_ct:
        found_key_out[0] = candidate_key

def run_brute_force(target_ct, pt, known_key, known_bits):
    bits_to_brute = 64 - known_bits
    total_combinations = 1 << bits_to_brute
    
    # Setup GPU memory
    z_gpu = cuda.to_device(Z0_SEQ)
    found_key_out = cuda.to_device(np.zeros(1, dtype=np.uint64))
    
    # RTX 5050 Optimization: Balanced block size
    threads_per_block = 256
    blocks_per_grid = (total_combinations + (threads_per_block - 1)) // threads_per_block
    
    print(f"Brute forcing {bits_to_brute} bits...")
    print(f"Dispatching {blocks_per_grid} blocks...")

    brute_force_kernel[blocks_per_grid, threads_per_block](
        uint32(target_ct), uint32(pt), uint64(known_key), 
        bits_to_brute, z_gpu, found_key_out
    )
    
    cuda.synchronize()
    result = found_key_out.copy_to_host()[0]
    return result

# --- Test Case ---
# Known Key: 0x1918111009080100
# Known 48 bits: 0x191811100908, Missing: 0x0100 (16 bits)
#TARGET_CIPHERTEXT = 0xb35e390c
TARGET_CIPHERTEXT = 0xc69be9bb 
PLAINTEXT = 0x65656877
KNOWN_PART = 0x191811100908 
KNOWN_LEN = 48

found = run_brute_force(TARGET_CIPHERTEXT, PLAINTEXT, KNOWN_PART, KNOWN_LEN)

if found != 0:
    print(f"SUCCESS! Key Found: {hex(found)}")
else:
    print("FAILURE: Key not found.")
