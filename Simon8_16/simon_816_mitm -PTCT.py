"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Correlated-Sequence MitM Attack on Simon-8/16  —  Pure Attacker Model     ║
║   n = 4 bits · 10 reduced rounds · 16-bit key recovery from PT-CT pairs     ║
╚══════════════════════════════════════════════════════════════════════════════╝

What this program does
----------------------
The user supplies only plaintext-ciphertext (PT-CT) pairs, exactly as an
attacker would receive them from an oracle (e.g. a captured device).
No key is given as input. The program finds the 16-bit key [k0,k1,k2,k3]
using the correlated-sequence Meet-in-the-Middle algorithm.

Three modes
-----------
  1. Interactive  — run with no arguments; menu asks for PT-CT pairs one by one
  2. File input   — --pairs-file pairs.txt  (one  "L R  L R"  per line)
  3. Quick test   — --demo  uses the built-in verified example

Usage examples
--------------
  python simon_816_mitm.py
      → interactive: enter number of pairs then each pair manually

  python simon_816_mitm.py --demo
      → uses built-in pairs (PT=(7,3) etc.) and shows full step-by-step trace

  python simon_816_mitm.py --pairs-file my_pairs.txt
      → reads pairs from a text file (format: "7 3  6 F" per line)

  python simon_816_mitm.py --pairs-file my_pairs.txt --quiet
      → silent run, prints only the recovered key

Pairs file format (--pairs-file)
---------------------------------
  # lines starting with # are comments
  7 3  6 F          <- PT_L PT_R  CT_L CT_R  (hex nibbles, space-separated)
  F 5  5 B
  A 6  C 8

Cipher
------
  Simon-8/16  n=4  (a,b,c)=(0,1,2)  10 reduced rounds  key=16 bits
  f(x) = (x & rl(x,1)) ^ rl(x,2)
  Round: s[i+2] = FT[s[i+1]] ^ s[i] ^ rk[i]

MitM round split
----------------
  Enc window  T_ENC  = 3   rounds 0,1,2   → produces se3, Xe
  Bridge      PARTIAL = 4  rounds 3,4,5,6  → enc_meet = s[7]
  Dec window  T_DEC  = 3   rounds 7,8,9   → sd_meet  from DS^d
  Match: enc_meet == sd_meet  at state s[7]
"""

import argparse
import sys
import time
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CIPHER CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

N       = 4
MASK    = 0xF
ROUNDS  = 10
T_ENC   = 3
T_DEC   = 3
PARTIAL = ROUNDS - T_ENC - T_DEC   # = 4

A, B, C = 0, 1, 2

Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,
      1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,
      1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]


def rl(x, r):
    r &= (N - 1)
    return ((x << r) | (x >> (N - r))) & MASK


def simon_f(x):
    return (rl(x, A) & rl(x, B)) ^ rl(x, C)


#   x:       0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
#   FT[x]:   0  4  8  E  1  5  D  B  2  7  A  D  B  E  7  0
FT = [simon_f(x) for x in range(1 << N)]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CIPHER PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def key_schedule(k, rounds=ROUNDS):
    CON = (MASK - 3) & MASK
    rk  = list(k)
    for i in range(rounds - 4):
        rr3 = ((rk[i+3] >> 3) | (rk[i+3] << (N-3))) & MASK
        rr4 = ((rk[i+3] >> 4) | (rk[i+3] << (N-4))) & MASK
        rr1 = ((rk[i+1] >> 1) | (rk[i+1] << (N-1))) & MASK
        t   = rr3 ^ rr4 ^ rk[i+1] ^ rr1
        rk.append((CON ^ Z0[i % 62] ^ rk[i] ^ t) & MASK)
    return rk[:rounds]


def encrypt_block(s0, s1, rk):
    L, R = s0, s1
    for i in range(len(rk)):
        L, R = R, (L ^ FT[R] ^ rk[i]) & MASK
    return L, R


def nlfsr_trace(s0, s1, rk):
    s = [s0, s1]
    for i in range(len(rk)):
        s.append((FT[s[-1]] ^ s[-2] ^ rk[i]) & MASK)
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COSET / Z-SEGMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def coset_leader(x):
    mn = x; cur = x
    for _ in range(N - 1):
        cur = rl(cur, 1)
        if cur < mn: mn = cur
    return mn


def build_z_segments():
    """
    Return (Z_vals, CLZ):
      Z_vals = sorted list of distinct z-values  (Nz = 4 for n=4)
      CLZ[i] = list of coset leaders with z(s) = Z_vals[i]
    """
    leaders = [x for x in range(1 << N) if coset_leader(x) == x]
    z_map   = defaultdict(list)
    for s in leaders:
        z = (FT[s] ^ rl(s, C)) & MASK
        z_map[z].append(s)
    Z_vals = sorted(z_map.keys())
    return Z_vals, [z_map[z] for z in Z_vals]


def orbit_members(leader):
    members = []
    cur = leader
    for _ in range(N):
        if cur not in members: members.append(cur)
        cur = rl(cur, 1)
        if cur == leader: break
    return members


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INDEX SET  I(ke0, ke1)
# ═══════════════════════════════════════════════════════════════════════════════

def build_index_set(se3, Xe):
    """
    I[ke2] = FT[Xe ^ ke2] ^ se3
    = the ke3 that forces s5 = 0 when master key uses (ke0,ke1,ke2,ke3).
    For any target s5=y: ke3 = I[ke2] ^ y  (Theorem 1, zero extra f-calls).
    """
    return [(FT[(Xe ^ ke2) & MASK] ^ se3) & MASK for ke2 in range(1 << N)]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DS^d  OFFLINE TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_dsd(ct0, ct1):
    """
    DS^d[kd0][kd1] = (sd2, sd3)
    sd0 = CT[1] = ct1  (= s[r+1])
    sd1 = CT[0] = ct0  (= s[r])
    sd2 = FT[sd1] ^ sd0 ^ kd0
    sd3 = FT[sd2] ^ sd1 ^ kd1
    """
    sd0, sd1 = ct1, ct0
    DS2 = [[0]*16 for _ in range(16)]
    DS3 = [[0]*16 for _ in range(16)]
    for kd0 in range(16):
        for kd1 in range(16):
            sd2 = (FT[sd1] ^ sd0 ^ kd0) & MASK
            sd3 = (FT[sd2] ^ sd1 ^ kd1) & MASK
            DS2[kd0][kd1] = sd2
            DS3[kd0][kd1] = sd3
    return DS2, DS3


def compute_sd_meet(rk, DS2, DS3):
    """
    sd_meet = FT[sd3] ^ sd2 ^ kd2
    kd0=rk[9], kd1=rk[8], kd2=rk[7]
    """
    kd0 = rk[ROUNDS-1]
    kd1 = rk[ROUNDS-2]
    kd2 = rk[ROUNDS-T_DEC]
    sd2 = DS2[kd0][kd1]
    sd3 = DS3[kd0][kd1]
    return (FT[sd3] ^ sd2 ^ kd2) & MASK


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FILTER KEYS  (Algorithm 2)
# ═══════════════════════════════════════════════════════════════════════════════

def filter_keys(se0, se1, DS2, DS3):
    """
    Scan all 16^4 key candidates.
    For each (ke0,ke1): pay 3+Nz f-calls once, covering all (ke2,ke3).
    For each (ke2,ke3): pay PARTIAL=4 f-calls for partial encryption.
    Return all candidates where enc_meet == sd_meet.
    """
    candidates = []
    for ke0 in range(1 << N):
        se2 = (FT[se1] ^ se0 ^ ke0) & MASK
        for ke1 in range(1 << N):
            se3 = (FT[se2] ^ se1 ^ ke1) & MASK
            Xe  = (FT[se3] ^ se2) & MASK
            Ie  = build_index_set(se3, Xe)
            for ke2 in range(1 << N):
                se4 = (Xe ^ ke2) & MASK
                for ke3 in range(1 << N):
                    rk = key_schedule([ke0, ke1, ke2, ke3])
                    L, R = se3, se4
                    for rnd in range(T_ENC, T_ENC + PARTIAL):
                        L, R = R, (L ^ FT[R] ^ rk[rnd]) & MASK
                    if L == compute_sd_meet(rk, DS2, DS3):
                        candidates.append([ke0, ke1, ke2, ke3])
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — KEY RECOVERY  (Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════════════

def recover_key(known_pairs, verbose=False):
    """
    Recover the 16-bit master key from PT-CT pairs alone.
    No key material is used as input — pure attacker model.

    Parameters
    ----------
    known_pairs : list of >= 3  ((pt_L,pt_R),(ct_L,ct_R)) tuples
    verbose     : print step-by-step progress

    Returns
    -------
    [k0,k1,k2,k3] on success, or None
    """
    assert len(known_pairs) >= 3, "Need at least 3 PT-CT pairs"

    (se0, se1), (ct0, ct1) = known_pairs[0]

    # ── OFFLINE ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    DS2, DS3 = build_dsd(ct0, ct1)
    t_off = time.perf_counter() - t0

    if verbose:
        print(f"\n  [offline] DS^d built in {t_off:.4f}s"
              f"  sd0={ct1:X}  sd1={ct0:X}")

    # ── ONLINE ─────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    candidates = filter_keys(se0, se1, DS2, DS3)
    t_online = time.perf_counter() - t1

    if verbose:
        print(f"  [online]  filter_keys: {len(candidates)} candidates"
              f"  in {t_online:.4f}s")

    # ── VERIFY ─────────────────────────────────────────────────────────────
    for (p0, p1), (c0, c1) in known_pairs[1:]:
        candidates = [k for k in candidates
                      if encrypt_block(p0, p1, key_schedule(k)) == (c0, c1)]

    if verbose:
        print(f"  [verify]  after verification: {len(candidates)} candidate(s)")
        print(f"  [total]   {time.perf_counter()-t0:.4f}s")

    return candidates[0] if candidates else None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — INPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_nibble(s):
    """Parse a hex nibble string like '7', 'F', '0xA' → int."""
    v = int(s, 16) if s.startswith('0x') or s.startswith('0X') \
        else int(s, 16) if all(c in '0123456789ABCDEFabcdef' for c in s) \
        else int(s)
    return v & MASK


def read_pairs_interactive():
    """
    Prompt the user for PT-CT pairs one by one.
    Returns list of ((pt_L,pt_R),(ct_L,ct_R)).
    """
    print()
    print("  Enter PT-CT pairs as hex nibbles (0-F).")
    print("  Example: PT = 7 3  CT = 6 F  → type: 7 3 6 F")
    print()

    while True:
        try:
            n = int(input("  How many PT-CT pairs do you have? (minimum 3): "))
            if n >= 3:
                break
            print("  ✗  Need at least 3 pairs for the attack.")
        except ValueError:
            print("  ✗  Please enter a whole number.")

    pairs = []
    for i in range(n):
        while True:
            try:
                raw = input(f"\n  Pair {i+1}/{n}  [PT_L  PT_R  CT_L  CT_R]: ").split()
                if len(raw) != 4:
                    raise ValueError("Need exactly 4 values")
                pt_l, pt_r, ct_l, ct_r = [parse_nibble(x) for x in raw]
                pairs.append(((pt_l, pt_r), (ct_l, ct_r)))
                print(f"           PT=({pt_l:X},{pt_r:X})  CT=({ct_l:X},{ct_r:X})  ✓")
                break
            except Exception as e:
                print(f"  ✗  Bad input ({e}). Enter 4 hex nibbles, e.g.: 7 3 6 F")
    return pairs


def read_pairs_file(path):
    """
    Read PT-CT pairs from a text file.
    Format per line: PT_L PT_R  CT_L CT_R  (hex, spaces, # comments ignored)
    """
    pairs = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                print(f"  ✗  Line {lineno} skipped (expected 4 values): {raw.rstrip()}")
                continue
            try:
                pt_l, pt_r, ct_l, ct_r = [parse_nibble(x) for x in parts]
                pairs.append(((pt_l, pt_r), (ct_l, ct_r)))
            except Exception as e:
                print(f"  ✗  Line {lineno} skipped ({e}): {raw.rstrip()}")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

SEP = "=" * 66
DIV = "─" * 66


def print_header():
    print()
    print(SEP)
    print("  Simon-8/16 · Correlated-Sequence MitM · Key Recovery")
    print("  n=4 bits · 16-bit key · 10 reduced rounds")
    print(SEP)


def print_ftable():
    print()
    print("  f-table  f(x) = (x & rl(x,1)) ^ rl(x,2)")
    print("  x    : " + "  ".join(f"{x:X}" for x in range(16)))
    print("  FT[x]: " + "  ".join(f"{FT[x]:X}" for x in range(16)))


def print_pairs(pairs):
    print()
    print(f"  Input: {len(pairs)} PT-CT pair(s)")
    print("  " + DIV)
    print(f"  {'#':>3}   {'PT (L,R)':>10}   {'CT (L,R)':>10}   Note")
    print("  " + DIV)
    for i, ((pl, pr), (cl, cr)) in enumerate(pairs):
        note = "← used for DS^d build + filter" if i == 0 \
               else "← used for verification"
        print(f"  {i+1:>3}   ({pl:X},{pr:X})           ({cl:X},{cr:X})       {note}")
    print()


def print_z_segments():
    Z_vals, CLZ = build_z_segments()
    print()
    print(f"  Z-linear segment sets (Nz={len(Z_vals)}, n={N})")
    print("  " + DIV)
    for z, ldr in zip(Z_vals, CLZ):
        members = sorted(set(
            m for s in ldr for m in orbit_members(s)
        ))
        fcall = "← 1 f-call" if ldr else ""
        print(f"   z={z:X}: leaders={str([hex(x) for x in ldr]):20s}"
              f"  covers={[hex(x) for x in members]}")
    print(f"\n  Index-set cost per (ke0,ke1): 3 (Step A) + {len(Z_vals)} (Step B)"
          f" = {3+len(Z_vals)} f-calls")
    print(f"  Naive cost: 3 + 16 = 19.  Saving: "
          f"{(16-len(Z_vals))/16*100:.0f}%")


def print_attack_steps(pairs, verbose=True):
    """
    Run the attack and print each phase with detailed output.
    """
    (se0, se1), (ct0, ct1) = pairs[0]

    print()
    print(SEP)
    print("  ATTACK EXECUTION")
    print(SEP)

    # ── OFFLINE ────────────────────────────────────────────────────────────
    print()
    print("  PHASE 1 — OFFLINE: Build DS^d")
    print("  " + DIV)
    sd0, sd1 = ct1, ct0
    print(f"  SD starting state:  sd0 = CT[1] = {ct1:X}   (= s_{{r+1}})")
    print(f"                      sd1 = CT[0] = {ct0:X}   (= s_r)")
    print(f"  For each (kd0,kd1) in [0,F]² → compute sd2, sd3 and store")
    t0 = time.perf_counter()
    DS2, DS3 = build_dsd(ct0, ct1)
    print(f"  DS^d built in {time.perf_counter()-t0:.4f}s  "
          f"(16×16 = {16*16} entries, 2 f-calls each)")

    # Sample DS^d entry for illustration
    print(f"\n  Sample entry: DS^d[0][0]:")
    sd2_s = (FT[sd1] ^ sd0 ^ 0) & MASK
    sd3_s = (FT[sd2_s] ^ sd1 ^ 0) & MASK
    print(f"    sd2 = FT[{sd1:X}]^{sd0:X}^0 = {FT[sd1]:X}^{sd0:X}^0 = {sd2_s:X}")
    print(f"    sd3 = FT[{sd2_s:X}]^{sd1:X}^0 = {FT[sd2_s]:X}^{sd1:X}^0 = {sd3_s:X}")

    # ── ONLINE ─────────────────────────────────────────────────────────────
    print()
    print("  PHASE 2 — ONLINE: Filter keys via partial encryption")
    print("  " + DIV)
    print(f"  Using pair 1: PT=({se0:X},{se1:X})  CT=({ct0:X},{ct1:X})")
    print()
    print("  For each (ke0,ke1) in [0,F]²:")
    print("    Step A: 3 f-calls → se2, se3, Xe")
    print("    Step B: 4 f-calls (Nz) → index set I[0..F]")
    print("    For each (ke2,ke3): partial-enc 4 rounds → enc_meet")
    print("    Check:  enc_meet == sd_meet from DS^d")
    print()

    t1 = time.perf_counter()
    candidates = filter_keys(se0, se1, DS2, DS3)
    t_filt = time.perf_counter() - t1
    print(f"  Candidates after filter: {len(candidates):5d}"
          f"  (in {t_filt:.4f}s)")

    # ── VERIFY ─────────────────────────────────────────────────────────────
    print()
    print("  PHASE 3 — VERIFY candidates against remaining pairs")
    print("  " + DIV)
    survivors = list(candidates)
    for idx, ((p0, p1), (c0, c1)) in enumerate(pairs[1:], 2):
        before = len(survivors)
        survivors = [k for k in survivors
                     if encrypt_block(p0, p1, key_schedule(k)) == (c0, c1)]
        print(f"  Pair {idx}: PT=({p0:X},{p1:X}) CT=({c0:X},{c1:X})"
              f"  → {before:4d} candidates → {len(survivors):4d} remaining")

    # ── RESULT ─────────────────────────────────────────────────────────────
    print()
    print("  " + SEP)
    t_total = time.perf_counter() - t0
    if survivors:
        key = survivors[0]
        rk  = key_schedule(key)
        print(f"  ★  KEY RECOVERED in {t_total:.4f}s")
        print()
        print(f"  Key  : [{key[0]:X}, {key[1]:X}, {key[2]:X}, {key[3]:X}]  "
              f"(hex nibbles)")
        print(f"  Hex  : {key[0]:X}{key[1]:X}{key[2]:X}{key[3]:X}")
        print(f"  Dec  : [{key[0]}, {key[1]}, {key[2]}, {key[3]}]")
        print()
        print(f"  Subkeys rk[0..9]: {[hex(x) for x in rk]}")
        print()

        # Re-verify all supplied pairs
        print("  Final verification against all input pairs:")
        print("  " + DIV)
        all_ok = True
        for i, ((pl, pr), (cl, cr)) in enumerate(pairs, 1):
            ct_check = encrypt_block(pl, pr, rk)
            ok = (ct_check == (cl, cr))
            all_ok = all_ok and ok
            print(f"  Pair {i}: PT=({pl:X},{pr:X})"
                  f"  Enc→({ct_check[0]:X},{ct_check[1]:X})"
                  f"  Expected ({cl:X},{cr:X})"
                  f"  {'✓' if ok else '✗ MISMATCH'}")
        print()
        if all_ok:
            print("  ✓  All pairs verified.  Attack successful.")
        else:
            print("  ✗  Some pairs did not verify.  Something went wrong.")
    else:
        print(f"  ✗  No key found. Check your PT-CT pairs.")

    print("  " + SEP)
    return survivors[0] if survivors else None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DEMO (built-in verified example)
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_PAIRS = [
    ((0x7, 0x3), (0x6, 0xF)),   # pair 1 — primary
    ((0xF, 0x5), (0x5, 0xB)),   # pair 2 — verification
    ((0xA, 0x6), (0xC, 0x8)),   # pair 3 — verification
]
# These correspond to Simon-8/16 key = [1,2,3,4].
# The attacker does NOT know this — the attack finds it.


def run_demo():
    print_header()
    print_ftable()
    print_z_segments()
    print()
    print("  Demo mode: using built-in PT-CT pairs")
    print("  (Correct key is [1,2,3,4] — but the attack does not use this)")
    print_pairs(DEMO_PAIRS)
    print_attack_steps(DEMO_PAIRS, verbose=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_interactive():
    print_header()
    print_ftable()
    print()
    print("  You are the attacker.")
    print("  You have captured plaintext-ciphertext pairs from a Simon-8/16")
    print("  device running 10 rounds.  Enter the pairs below.")
    print("  The program will recover the 16-bit key.")
    print_z_segments()

    pairs = read_pairs_interactive()
    print_pairs(pairs)

    print()
    show_steps = input(
        "  Show detailed attack steps? [Y/n]: "
    ).strip().lower()
    verbose = (show_steps != 'n')

    if verbose:
        print_attack_steps(pairs, verbose=True)
    else:
        print()
        key = recover_key(pairs, verbose=True)
        if key:
            print(f"\n  ★  Key = [{key[0]:X},{key[1]:X},{key[2]:X},{key[3]:X}]")
        else:
            print("\n  ✗  Key not found.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — FILE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_file(path, quiet=False):
    pairs = read_pairs_file(path)
    if len(pairs) < 3:
        print(f"  ✗  Need at least 3 valid pairs; found {len(pairs)} in {path}")
        sys.exit(1)

    if not quiet:
        print_header()
        print_ftable()
        print_z_segments()
        print_pairs(pairs)
        key = print_attack_steps(pairs, verbose=True)
    else:
        key = recover_key(pairs, verbose=False)
        if key:
            print(f"[{key[0]:X},{key[1]:X},{key[2]:X},{key[3]:X}]")
        else:
            print("KEY_NOT_FOUND")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse():
    p = argparse.ArgumentParser(
        description="Simon-8/16 MitM key recovery from PT-CT pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
  Interactive (default, no arguments):
    python simon_816_mitm.py

  Demo with built-in pairs:
    python simon_816_mitm.py --demo

  Read pairs from a file:
    python simon_816_mitm.py --pairs-file pairs.txt

  Silent (key only, for scripting):
    python simon_816_mitm.py --pairs-file pairs.txt --quiet

Pairs file format (one pair per line, hex nibbles):
  # This is a comment
  7 3  6 F
  F 5  5 B
  A 6  C 8
        """,
    )
    p.add_argument("--demo",
                   action="store_true",
                   help="Run built-in verified example")
    p.add_argument("--pairs-file", metavar="FILE",
                   help="Read PT-CT pairs from a text file")
    p.add_argument("--quiet",
                   action="store_true",
                   help="Print only the recovered key (for scripting)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.demo:
        run_demo()
    elif args.pairs_file:
        run_file(args.pairs_file, quiet=args.quiet)
    else:
        run_interactive()
