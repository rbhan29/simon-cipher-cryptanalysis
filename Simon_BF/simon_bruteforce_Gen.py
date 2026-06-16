import numpy as np
from numba import cuda, uint16, uint32, uint64
import time

# SIMON 32/64 Constants
C_CONST = 0xFFFC
Z0_SEQ = np.array([1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,
                   1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0], dtype=np.uint16)

# Updated Target Pairs (Verified)
PT1, CT1 = 0xaaaaffff, 0x92ccf4ae
PT2, CT2 = 0x12345678, 0xf82af9d3
'''
# User Configuration
# key used : KEY = 0x1918111009080100
KNOWN_HEX = 0x1918  # Fixed MSB (16 bits)
KNOWN_LEN = 16
'''
'''
# User Configuration
KNOWN_HEX = 0x19181110  # Fixed MSB (32 bits)
KNOWN_LEN = 32
'''

# User Configuration
KNOWN_HEX = 0x191811  # Fixed MSB (24 bits)
KNOWN_LEN = 24

@cuda.jit(device=True)
def rotate_left(x, n):
    return ((x << n) & 0xFFFF) | (x >> (16 - n))

@cuda.jit(device=True)
def rotate_right(x, n):
    return (x >> n) | ((x << (16 - n)) & 0xFFFF)

@cuda.jit(device=True)
def simon_encrypt_core(pt, subkeys):
    x = uint16((pt >> 16) & 0xFFFF)
    y = uint16(pt & 0xFFFF)
    for i in range(32):
        f_x = (rotate_left(x, 1) & rotate_left(x, 8)) ^ rotate_left(x, 2)
        new_x = uint16(y ^ f_x ^ subkeys[i])
        y, x = x, new_x
    return (uint32(x) << 16) | uint32(y)

@cuda.jit
def brute_force_kernel(ct1, pt1, ct2, pt2, known_part, brute_bits, z_gpu, offset, found_key_out):
    idx = cuda.grid(1)
    current_val = uint64(idx) + uint64(offset)
    
    # Construct 64-bit key: [Known Part] [Brute Part]
    candidate_key = uint64((uint64(known_part) << brute_bits) | current_val)
    
    # Key Expansion (Proper Endianness: LS Word is candidate_key & 0xFFFF)
    k = cuda.local.array(32, dtype=uint16)
    for i in range(4):
        k[i] = uint16((candidate_key >> (16 * i)) & 0xFFFF)
    
    for i in range(4, 32):
        tmp = rotate_right(k[i-1], 3) ^ k[i-3]
        tmp ^= rotate_right(tmp, 1)
        k[i] = uint16(k[i-4] ^ tmp ^ z_gpu[(i-4) % 62] ^ C_CONST)

    # Multi-Pair Verification
    if simon_encrypt_core(pt1, k) == ct1:
        if simon_encrypt_core(pt2, k) == ct2:
            found_key_out[0] = candidate_key

def run_verified_brute():
    brute_bits = 64 - KNOWN_LEN
    total_iter = 1 << brute_bits
    gpu_batch_size = 1 << 26 
    
    z_gpu = cuda.to_device(Z0_SEQ)
    found_key_out = cuda.to_device(np.zeros(1, dtype=np.uint64))
    
    tp_block = 256
    bp_grid = gpu_batch_size // tp_block

    print(f"Brute forcing {brute_bits} bits...")
    start_time = time.time()
    print(start_time)

    for offset in range(0, total_iter, gpu_batch_size):
        brute_force_kernel[bp_grid, tp_block](
            uint32(CT1), uint32(PT1), uint32(CT2), uint32(PT2), 
            uint64(KNOWN_HEX), brute_bits, z_gpu, 
            uint64(offset), found_key_out
        )
        
        cuda.synchronize()
        res = found_key_out.copy_to_host()
        if res[0] != 0:
            print(f"\n[!] MASTER KEY FOUND: {hex(res[0])}")
            end_time=time.time()
            print(end_time)
            #elapsed_time = time.perf_counter() - start_time
            elapsed_time = end_time-start_time
            print(f"Total Time Taken: {elapsed_time:.4f} seconds")

            return res[0]
            
        progress = ((offset + gpu_batch_size) / total_iter) * 100
        print(f"Progress: {progress:.6f}%", end='\r')

    return None

if __name__ == "__main__":
    run_verified_brute()
