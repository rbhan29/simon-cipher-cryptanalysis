# Simon 32/64 Constants
C = 0xFFFC
Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,
      1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]

def rotate_left(x, n, bits=16):
    return ((x << n) & (2**bits - 1)) | (x >> (bits - n))

def rotate_right(x, n, bits=16):
    return (x >> n) | ((x << (bits - n)) & (2**bits - 1))

def get_subkeys(key_64, rounds):
    # Split 64-bit key into four 16-bit words (m=4)
    k = [(key_64 >> (16 * i)) & 0xFFFF for i in range(4)]
    for i in range(4, rounds):
        tmp = rotate_right(k[i-1], 3)
        tmp ^= k[i-3]  # Specific to m=4
        tmp ^= rotate_right(tmp, 1)
        next_k = (k[i-4] ^ tmp ^ Z0[(i-4) % 62] ^ C) & 0xFFFF
        k.append(next_k)
    return k
def simon_encrypt(plaintext_32, key_64, rounds=32):
    subkeys = get_subkeys(key_64, rounds)
    x = (plaintext_32 >> 16) & 0xFFFF
    y = plaintext_32 & 0xFFFF
    
    for i in range(rounds):
        tmp = x
        # Round function
        x = (y ^ (rotate_left(x, 1) & rotate_left(x, 8)) ^ rotate_left(x, 2) ^ subkeys[i]) & 0xFFFF
        y = tmp
    return (x << 16) | y

def simon_decrypt(ciphertext_32, key_64, rounds=32):
    subkeys = get_subkeys(key_64, rounds)
    x = (ciphertext_32 >> 16) & 0xFFFF
    y = ciphertext_32 & 0xFFFF
    
    # Apply subkeys in reverse
    for i in range(rounds - 1, -1, -1):
        tmp = y
        y = (x ^ (rotate_left(y, 1) & rotate_left(y, 8)) ^ rotate_left(y, 2) ^ subkeys[i]) & 0xFFFF
        x = tmp
    return (x << 16) | y
# Configuration
KEY = 0x1918111009080100
PT = 0xaaaaffff
USER_ROUNDS = 32

# Execute
cipher = simon_encrypt(PT, KEY, rounds=USER_ROUNDS)
plain = simon_decrypt(cipher, KEY, rounds=USER_ROUNDS)

print(f"Original: {hex(PT)}")
print(f"Ciphertext: {hex(cipher)}")
print(f"Decrypted: {hex(plain)}")
