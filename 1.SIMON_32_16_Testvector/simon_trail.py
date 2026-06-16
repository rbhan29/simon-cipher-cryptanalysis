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
    # k[0] is the least significant word
    k = [(key_64 >> (16 * i)) & 0xFFFF for i in range(4)]
    
    for i in range(4, rounds):
        tmp = rotate_right(k[i-1], 3)
        tmp ^= k[i-3]
        tmp ^= rotate_right(tmp, 1)
        next_k = (k[i-4] ^ tmp ^ Z0[(i-4) % 62] ^ C) & 0xFFFF
        k.append(next_k)
    return k

def simon_encrypt_trace(plaintext_32, key_64, rounds=32):
    subkeys = get_subkeys(key_64, rounds)
    x = (plaintext_32 >> 16) & 0xFFFF # Left word
    y = plaintext_32 & 0xFFFF         # Right word
    
    print(f"\n--- Encryption Trace ({rounds} rounds) ---")
    print(f"Round | Subkey | Left (X) | Right (Y)")
    print(f"---------------------------------------")
    
    for i in range(rounds):
        # Round function: x_{i+1} = y_i ^ f(x_i) ^ k_i, y_{i+1} = x_i
        f_x = (rotate_left(x, 1) & rotate_left(x, 8)) ^ rotate_left(x, 2)
        new_x = (y ^ f_x ^ subkeys[i]) & 0xFFFF
        y = x
        x = new_x
        print(f"{i:5} | {hex(subkeys[i]):6} | {hex(x):8} | {hex(y):9}")
        
    return (x << 16) | y

def simon_decrypt_trace(ciphertext_32, key_64, rounds=32):
    subkeys = get_subkeys(key_64, rounds)
    x = (ciphertext_32 >> 16) & 0xFFFF
    y = ciphertext_32 & 0xFFFF
    
    print(f"\n--- Decryption Trace ({rounds} rounds) ---")
    print(f"Round | Subkey | Left (X) | Right (Y)")
    print(f"---------------------------------------")
    
    for i in range(rounds - 1, -1, -1):
        # Inverse: y_i = x_{i+1} ^ f(y_{i+1}) ^ k_i, x_i = y_{i+1}
        f_y = (rotate_left(y, 1) & rotate_left(y, 8)) ^ rotate_left(y, 2)
        new_y = (x ^ f_y ^ subkeys[i]) & 0xFFFF
        x = y
        y = new_y
        print(f"{i:5} | {hex(subkeys[i]):6} | {hex(x):8} | {hex(y):9}")
        
    return (x << 16) | y

# --- Configuration ---
KEY = 0x1918111009080100
PT = 0x65656877
USER_ROUNDS = 32 # You can change this value

# --- Execution ---
cipher = simon_encrypt_trace(PT, KEY, rounds=USER_ROUNDS)
plain = simon_decrypt_trace(cipher, KEY, rounds=USER_ROUNDS)

print("\n" + "="*30)
print(f"Original PT:  {hex(PT)}")
print(f"Ciphertext:   {hex(cipher)}")
print(f"Decrypted PT: {hex(plain)}")
print("="*30)
