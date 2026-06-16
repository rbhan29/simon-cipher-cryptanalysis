# =============================================================================
#  Simon-8/16  —  Test Vector Generator
#  Set your parameters in the USER SETTINGS block below, then run:
#      python simon816_testvectors.py
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ★  USER SETTINGS  —  edit these three blocks, nothing else needs changing
# ─────────────────────────────────────────────────────────────────────────────

# Key: four 4-bit nibbles, each in range 0-15 (or 0x0-0xF)
#KEY = [1, 2, 3, 4]
KEY = [0xa,0xb,0xc,0xd]

# Plaintext pairs to encrypt: each entry is (Left_nibble, Right_nibble)
PLAINTEXTS = [
    (0x7, 0x3),
    (0xF, 0x5),
    (0xA, 0x6),
    (0x3, 0xC),
    (0x0, 0x0),
    (0xF, 0xF),
]

# Number of rounds (1–32; full Simon-8/16 = 32)
ROUNDS = 10

# ─────────────────────────────────────────────────────────────────────────────
#  CIPHER  (do not edit below this line)
# ─────────────────────────────────────────────────────────────────────────────

N    = 4
MASK = 0xF
Z0   = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,
        1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,
        1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]

def _rl(x, r):
    r &= (N - 1)
    return ((x << r) | (x >> (N - r))) & MASK

def _f(x):
    return (_rl(x, 0) & _rl(x, 1)) ^ _rl(x, 2)

FT = [_f(x) for x in range(16)]

def _key_schedule(k, rounds):
    CON = (MASK - 3) & MASK
    rk  = list(k)
    for i in range(rounds - 4):
        t = (((rk[i+3]>>3)|(rk[i+3]<<1))&MASK ^
             ((rk[i+3]>>0)|(rk[i+3]<<0))&MASK ^
               rk[i+1] ^
             ((rk[i+1]>>1)|(rk[i+1]<<3))&MASK)
        rk.append((CON ^ Z0[i % 62] ^ rk[i] ^ t) & MASK)
    return rk[:rounds]

def _encrypt(L, R, rk):
    for i in range(len(rk)):
        L, R = R, (L ^ FT[R] ^ rk[i]) & MASK
    return L, R

# ─────────────────────────────────────────────────────────────────────────────
#  GENERATE AND PRINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Validate
    assert len(KEY) == 4 and all(0 <= k <= 15 for k in KEY), \
        "KEY must be 4 values each in 0-15"
    assert 1 <= ROUNDS <= 32, "ROUNDS must be between 1 and 32"

    rk = _key_schedule(KEY, ROUNDS)

    print(f"Simon-8/16  |  Key: [{', '.join(hex(k) for k in KEY)}]"
          f"  |  Rounds: {ROUNDS}")
    print(f"Subkeys rk[0..{ROUNDS-1}]: {[hex(x) for x in rk]}")
    print()
    print(f"  {'#':>3}   {'PT (L, R)':>10}   {'CT (L, R)':>10}   {'Verify':>7}")
    print("  " + "-" * 42)

    pairs = []
    for i, (pl, pr) in enumerate(PLAINTEXTS, 1):
        pl, pr = pl & MASK, pr & MASK
        cl, cr = _encrypt(pl, pr, rk)

        # Verify: decrypt and check
        L, R = cl, cr
        for j in reversed(range(ROUNDS)):
            L, R = (FT[L] ^ R ^ rk[j]) & MASK, L
        ok = (L == pl and R == pr)

        print(f"  {i:>3}   ({pl:X}, {pr:X})           "
              f"({cl:X}, {cr:X})        {'✓' if ok else '✗'}")
        pairs.append(((pl, pr), (cl, cr)))

    print()
    print("PT-CT pairs (ready to copy into attack code):")
    print("  pairs = [")
    for (pl, pr), (cl, cr) in pairs:
        print(f"      (({pl:#x}, {pr:#x}), ({cl:#x}, {cr:#x})),")
    print("  ]")

if __name__ == "__main__":
    main()
