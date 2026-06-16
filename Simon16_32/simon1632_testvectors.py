# =============================================================================
#  Simon-16/32  —  Test Vector Generator
#  Set your parameters in the USER SETTINGS block below, then run:
#      python simon1632_testvectors.py
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ★  USER SETTINGS  —  edit these three blocks, nothing else needs changing
# ─────────────────────────────────────────────────────────────────────────────

# Key: four 16-bit words, each in range 0x0000 – 0xFFFF
#KEY = [0x0001, 0x0002, 0x0003, 0x0004]
KEY = [0x1001, 0x2002, 0x3003, 0x4004]

# Plaintext pairs to encrypt: each entry is (Left_word, Right_word), 16-bit
PLAINTEXTS = [
    (0x7FFF, 0x3FFF),
    (0x0F0E, 0x0000),
    (0xABCD, 0x1234),
    (0xFFFF, 0x5555),
    (0x0000, 0x0001),
    (0x1234, 0xABCD),
]

# Number of rounds (1–32; full Simon-16/32 = 32)
ROUNDS = 10

# ─────────────────────────────────────────────────────────────────────────────
#  CIPHER  (do not edit below this line)
# ─────────────────────────────────────────────────────────────────────────────

# Simon-16/32 parameters
_N    = 16           # word size in bits
_MASK = 0xFFFF       # 2^16 - 1
_A, _B, _C = 8, 1, 2 # shift params: f(x) = (x<<<8 & x<<<1) ^ x<<<2

# Key-schedule LFSR constant z0 (period 62, Simon specification)
_Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,
       1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,
       1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]


def _rl(x, r):
    """Cyclic left-rotate a 16-bit word by r positions."""
    r &= (_N - 1)
    return ((x << r) | (x >> (_N - r))) & _MASK


def _simon_f(x):
    """Simon nonlinear function f(x) = (x<<<8 & x<<<1) ^ x<<<2."""
    return (_rl(x, _A) & _rl(x, _B)) ^ _rl(x, _C)


def _key_schedule(k, rounds):
    """Expand 4-word master key to `rounds` subkeys."""
    CON = (_MASK - 3) & _MASK          # 0xFFFC
    rk  = list(k)
    for i in range(rounds - 4):
        t = (((rk[i+3] >> 3) | (rk[i+3] << (_N-3))) & _MASK ^
             ((rk[i+3] >> 4) | (rk[i+3] << (_N-4))) & _MASK ^
               rk[i+1] ^
             ((rk[i+1] >> 1) | (rk[i+1] << (_N-1))) & _MASK)
        rk.append((CON ^ _Z0[i % 62] ^ rk[i] ^ t) & _MASK)
    return rk[:rounds]


def _encrypt(L, R, rk):
    """Encrypt (L, R) for len(rk) rounds."""
    for i in range(len(rk)):
        L, R = R, (L ^ _simon_f(R) ^ rk[i]) & _MASK
    return L, R


def _decrypt(L, R, rk):
    """Decrypt (L, R) by reversing the Feistel steps."""
    for i in reversed(range(len(rk))):
        L, R = (_simon_f(L) ^ R ^ rk[i]) & _MASK, L
    return L, R


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATE AND PRINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Validate settings
    assert len(KEY) == 4 and all(0 <= k <= 0xFFFF for k in KEY), \
        "KEY must be 4 values each in 0x0000–0xFFFF"
    assert 1 <= ROUNDS <= 32, "ROUNDS must be between 1 and 32"

    rk = _key_schedule(KEY, ROUNDS)

    # Header
    print(f"Simon-16/32  |  "
          f"Key: [{', '.join(hex(k) for k in KEY)}]  |  Rounds: {ROUNDS}")
    print(f"Subkeys rk[0..{ROUNDS-1}]: {[hex(x) for x in rk]}")
    print()
    print(f"  {'#':>3}   {'PT  (L,    R)':>18}   {'CT  (L,    R)':>18}   {'Verify':>7}")
    print("  " + "-" * 58)

    pairs = []
    for i, (pl, pr) in enumerate(PLAINTEXTS, 1):
        pl, pr = pl & _MASK, pr & _MASK
        cl, cr = _encrypt(pl, pr, rk)

        # Decrypt and verify
        bl, br = _decrypt(cl, cr, rk)
        ok = (bl == pl and br == pr)

        print(f"  {i:>3}   (0x{pl:04X}, 0x{pr:04X})     "
              f"(0x{cl:04X}, 0x{cr:04X})        {'✓' if ok else '✗'}")
        pairs.append(((pl, pr), (cl, cr)))

    # Copy-paste block for attack code
    print()
    print("PT-CT pairs (ready to copy into attack code):")
    print("  pairs = [")
    for (pl, pr), (cl, cr) in pairs:
        print(f"      ((0x{pl:04x}, 0x{pr:04x}), (0x{cl:04x}, 0x{cr:04x})),")
    print("  ]")


if __name__ == "__main__":
    main()
