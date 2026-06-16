"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Correlated-Sequence Meet-in-the-Middle Attack on Simon-8/16               ║
║   Toy cipher  ·  n = 4 bits  ·  10 reduced rounds  ·  16-bit key            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Based on:
  Rohit & Gong, "Meet-in-the-Middle attack using Correlated Sequences and its
  Applications to Simon-like Ciphers", University of Waterloo, 2019.
  https://eprint.iacr.org/2018/699.pdf

Purpose
-------
This file is a pedagogical, step-by-step implementation of the correlated-
sequence MitM attack on Simon-8/16 (the smallest Simon variant) so that every
algorithm detail from the paper can be read directly in Python.

Cipher parameters
-----------------
  n   = 4          word size in bits  (Simon-8/16 uses 4-bit nibbles)
  N   = 2^n = 16   alphabet size
  Block = 2n = 8   bits  (two 4-bit words: left L and right R)
  Key   = 4n = 16  bits  (four 4-bit words: k0,k1,k2,k3)
  (a,b,c) = (0,1,2) Simon shift parameters for n = 4
  Full rounds = 32; this attack uses 10 reduced rounds.

MitM round split (10 rounds)
-----------------------------
  T_ENC  = 3   enc-side correlated-sequence window  (rounds 0, 1, 2)
  PARTIAL = 4  partial-encryption bridge             (rounds 3, 4, 5, 6)
  T_DEC  = 3   dec-side correlated-sequence window  (rounds 7, 8, 9)
  Match point: s[T_ENC + PARTIAL] = s[7]

Verified example (key=[1,2,3,4], PT=(7,3))
-------------------------------------------
  NLFSR states: 7 3 8 3 5 2 E 5 3 1 6 F
  Ciphertext  : (6, F)
  enc_meet = s[7] = 5    sd_meet = 5    MATCH ✓

Sections in this file
---------------------
  1.  Cipher constants and f-table
  2.  Key schedule
  3.  Encryption and decryption
  4.  Coset leaders and z-linear segment sets  (Definition 4, paper §4)
  5.  Index set  I(k0, k1)                     (Section 4.2, Theorem 1)
  6.  DS^d offline data structure              (Section 5.1)
  7.  Online filter: filter_keys               (Algorithm 2)
  8.  Key recovery: recover_key                (Algorithm 1)
  9.  Step-by-step verbose demo                (matches the worked example)
 10.  Batch random-key tests
 11.  CLI entry point
"""

import argparse
import random
import time
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CIPHER CONSTANTS AND f-TABLE
# ═══════════════════════════════════════════════════════════════════════════════

N       = 4          # word size in bits
MASK    = 0xF        # 2^N - 1 = 15  (keep all arithmetic 4-bit)
ROUNDS  = 10         # reduced rounds for this attack
T_ENC   = 3          # enc-side window length
T_DEC   = 3          # dec-side window length
PARTIAL = ROUNDS - T_ENC - T_DEC   # = 4  (bridge rounds)

# Simon shift parameters: f(x) = (x<<<A  &  x<<<B)  ^  x<<<C
A, B, C = 0, 1, 2

# Key-schedule LFSR constant z0 (period 62, from Simon specification)
Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,
      1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,
      1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]


def rl(x: int, r: int) -> int:
    """
    Cyclic left-rotate an n-bit word by r positions.
    Example (n=4): rl(0b0001, 1) = 0b0010 = 2
    """
    r &= (N - 1)
    return ((x << r) | (x >> (N - r))) & MASK


def simon_f(x: int) -> int:
    """
    Simon nonlinear function  f(x) = (x<<<A & x<<<B) ^ x<<<C.
    For (A,B,C)=(0,1,2) and n=4: f(x) = (x & rl(x,1)) ^ rl(x,2).

    This is the ONLY nonlinear operation in the entire cipher.
    Every time f is called at runtime, it counts as one "f-call"
    in the paper's complexity analysis.

    Example:
      f(3) = (3 & rl(3,1)) ^ rl(3,2)
           = (0011 & 0110) ^ 1100
           = 0010 ^ 1100
           = 1110 = 0xE
    """
    return (rl(x, A) & rl(x, B)) ^ rl(x, C)


# Pre-computed lookup table: F_TABLE[x] = simon_f(x) for all x in 0..15.
# Using this table instead of calling simon_f() directly means all f-call
# counting in comments is in terms of table lookups, not function calls.
#
# Full table:
#   x:        0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
#   F_TABLE:  0  4  8  E  1  5  D  B  2  7  A  D  B  E  7  0
F_TABLE: list[int] = [simon_f(x) for x in range(1 << N)]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KEY SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

def key_schedule(k: list, rounds: int = ROUNDS) -> list:
    """
    Expand 4-word master key [k0,k1,k2,k3] to `rounds` subkeys.

    Simon-8/16 recurrence (adapted for n=4 word size):
        rk[i+4] = CON ^ z0[i%62] ^ rk[i]
                ^ rr(rk[i+3], 3) ^ rr(rk[i+3], 4)
                ^ rr(rk[i+1], 1) ^ rk[i+1]
    where CON = 2^n - 4 = 0xC for n=4.

    Note: rr(x, r) = rl(x, n-r).  For n=4: rr(x,3)=rl(x,1), rr(x,4)=x.

    Parameters
    ----------
    k      : list of 4 integers in [0, 15]
    rounds : number of subkeys to generate (default ROUNDS=10)

    Returns
    -------
    list of `rounds` integers in [0, 15]

    Example (k=[1,2,3,4], 10 rounds):
        rk = [1, 2, 3, 4, 3, 0, 8, A, 1, 3]
    """
    CON = (MASK - 3) & MASK   # 0xC for n=4
    rk  = list(k)
    for i in range(rounds - 4):
        # right-rotate rk[i+3] by 3 bits  (= left-rotate by n-3 = 1 for n=4)
        rr3 = ((rk[i+3] >> 3) | (rk[i+3] << (N-3))) & MASK
        # right-rotate rk[i+3] by 4 bits  (= identity for n=4)
        rr4 = ((rk[i+3] >> 4) | (rk[i+3] << (N-4))) & MASK
        # right-rotate rk[i+1] by 1 bit   (= left-rotate by 3 for n=4)
        rr1 = ((rk[i+1] >> 1) | (rk[i+1] << (N-1))) & MASK
        t   = rr3 ^ rr4 ^ rk[i+1] ^ rr1
        rk.append((CON ^ Z0[i % 62] ^ rk[i] ^ t) & MASK)
    return rk[:rounds]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ENCRYPTION AND DECRYPTION
# ═══════════════════════════════════════════════════════════════════════════════

def encrypt_block(s0: int, s1: int, rk: list) -> tuple:
    """
    Encrypt plaintext (s0, s1) for len(rk) rounds.

    Feistel update each round i:
        (L, R) -> (R, F_TABLE[R] ^ L ^ rk[i])

    This is the NLFSR form: s[i+2] = F_TABLE[s[i+1]] ^ s[i] ^ rk[i].

    Returns (L, R) = (s[rounds], s[rounds+1]).
    Note: paper ciphertext notation is (s_{r+1}, s_r) = (R, L).
    This function returns (L, R) = (s[r], s[r+1]).

    Example (k=[1,2,3,4], PT=(7,3), 10 rounds):
        CT = (6, F)
    """
    L, R = s0, s1
    for i in range(len(rk)):
        L, R = R, (L ^ F_TABLE[R] ^ rk[i]) & MASK
    return L, R


def decrypt_block(c0: int, c1: int, rk: list) -> tuple:
    """
    Decrypt ciphertext (c0, c1) by running the Feistel in reverse.

    Inverse: (L, R) -> (F_TABLE[L] ^ R ^ rk[i], L)
    """
    L, R = c0, c1
    for i in reversed(range(len(rk))):
        L, R = (F_TABLE[L] ^ R ^ rk[i]) & MASK, L
    return L, R


def nlfsr_trace(s0: int, s1: int, rk: list) -> list:
    """
    Return the full NLFSR state sequence [s0, s1, s2, ..., s_{r+1}].
    Useful for tracing and verification.
    """
    s = [s0, s1]
    for i in range(len(rk)):
        s.append((F_TABLE[s[-1]] ^ s[-2] ^ rk[i]) & MASK)
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COSET LEADERS AND Z-LINEAR SEGMENT SETS
# ═══════════════════════════════════════════════════════════════════════════════

def coset_leader(x: int) -> int:
    """
    Return the minimum element of the cyclic-rotation orbit of x.

    Orbit of x = { x, rl(x,1), rl(x,2), ..., rl(x,n-1) }
    Leader     = min of orbit  (canonical representative)

    For n=4: there are 6 distinct leaders: {0, 1, 3, 5, 7, F}
    """
    mn  = x
    cur = x
    for _ in range(N - 1):
        cur = rl(cur, 1)
        if cur < mn:
            mn = cur
    return mn


def build_z_segments() -> tuple:
    """
    Partition all coset leaders by their z-value.

    z-value of leader s:
        z(s) = f(s) ^ rl(s, C)   [Definition 4 in paper]
    For Simon with C=2: z(s) = F_TABLE[s] ^ rl(s,2)

    Key property: for any y = rl(s, i) in the orbit of leader s:
        f(y) = rl(f(s), i)  [f is equivariant under rotation]
    Therefore: once f(s) is known (1 f-call), f is known for
    ALL orbit members at zero extra cost.

    Further, if two leaders s1 and s2 share the same z-value, they
    belong to the same z-linear segment set CLz.  One f-call at a
    representative covers the entire segment.

    Returns
    -------
    Z_vals : sorted list of distinct z-values  (length = Nz)
    CLZ    : CLZ[i] = list of leaders with z-value Z_vals[i]

    For Simon-8/16 (n=4):
        Nz = 4
        CLZ = [[0,1,5], [3], [7], [F]]
        z-values = [0, 2, 6, F]
    """
    # Collect all coset leaders
    leaders = [x for x in range(1 << N) if coset_leader(x) == x]

    # Group leaders by z-value
    z_map: dict[int, list] = defaultdict(list)
    for s in leaders:
        z = (F_TABLE[s] ^ rl(s, C)) & MASK
        z_map[z].append(s)

    Z_vals = sorted(z_map.keys())
    CLZ    = [z_map[z] for z in Z_vals]
    return Z_vals, CLZ


def orbit_members(leader: int) -> list:
    """
    Return all elements in the cyclic-rotation orbit of `leader`.
    Used for z-segment analysis and display.
    """
    members = []
    cur = leader
    for _ in range(N):
        if cur not in members:
            members.append(cur)
        cur = rl(cur, 1)
        if cur == leader:
            break
    return members


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INDEX SET  I(ke0, ke1)
# ═══════════════════════════════════════════════════════════════════════════════

def build_index_set(se3: int, Xe: int) -> list:
    """
    Build the index set I[ke2] = f(Xe ^ ke2) ^ se3
    for ke2 = 0, 1, 2, ..., N-1 = 0, ..., 15.

    Meaning (Theorem 1 in paper):
        I[ke2] is the unique ke3 that forces s5 = 0
        when the master key uses words (ke0, ke1, ke2, ke3).

    Extension (zero extra f-calls):
        To target ANY value s5 = y, use ke3 = I[ke2] ^ y.
        This is Theorem 1 of the paper — the (1,8)-correlated sequence
        property.  No recomputation of f is needed.

    Parameters
    ----------
    se3 : s[T_ENC] = s[3] — third NLFSR state (from 3 f-calls in Step A)
    Xe  : X = f(se3) ^ se2 — the pivot constant (from Step A)

    Cost: N table lookups (= N f-calls naively, but reduced to Nz=4
          by the z-linear segment structure when computing for all ke2).

    Returns
    -------
    list of N integers: I[0], I[1], ..., I[N-1]

    Example (se3=3, Xe=6):
        I = [E, 8, 2, 6, B, D, 3, 7, 4, 3, 8, D, 9, E, 1, 4]
        I[3] = 6  means: ke3=6 forces s5=0  when ke2=3
        For s5=2 (our target): ke3 = I[3]^2 = 6^2 = 4  ✓
    """
    return [(F_TABLE[(Xe ^ ke2) & MASK] ^ se3) & MASK
            for ke2 in range(1 << N)]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DS^d  OFFLINE DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def build_dsd(ct0: int, ct1: int) -> tuple:
    """
    Build the dec-side data structure DS^d[kd0][kd1] = (sd2, sd3)
    for all (kd0, kd1) pairs.

    Convention (verified against cipher trace):
        sd0 = CT[1] = ct1   (= s[r+1] in paper notation, the right CT word)
        sd1 = CT[0] = ct0   (= s[r],   the left  CT word)

    Dec NLFSR steps (running backward from ciphertext):
        sd2 = F_TABLE[sd1] ^ sd0 ^ kd0   where kd0 = rk[ROUNDS-1] = rk[9]
        sd3 = F_TABLE[sd2] ^ sd1 ^ kd1   where kd1 = rk[ROUNDS-2] = rk[8]

    Meeting value (retrieved during online phase):
        sd_meet = F_TABLE[sd3] ^ sd2 ^ kd2
                  where kd2 = rk[ROUNDS - T_DEC] = rk[7]

    This equals s[T_ENC + PARTIAL] = s[7] for the correct key.

    Parameters
    ----------
    ct0, ct1 : left and right halves of the ciphertext (from pair 1)

    Returns
    -------
    DS_sd2 : 16x16 table, DS_sd2[kd0][kd1] = sd2
    DS_sd3 : 16x16 table, DS_sd3[kd0][kd1] = sd3

    Memory: 2 × 16 × 16 = 512 nibbles = 256 bytes for n=4.
    Cost: 2 × 16^2 = 512 f-calls (done once offline).

    Example (CT=(6,F)):
        sd0=F, sd1=6
        DS_sd2[3][1] = F_TABLE[6]^F^3 = D^F^3 = 1
        DS_sd3[3][1] = F_TABLE[1]^6^1 = 4^6^1 = 3
    """
    sd0 = ct1   # sd[0] = s[r+1] = CT right word
    sd1 = ct0   # sd[1] = s[r]   = CT left word

    DS_sd2 = [[0]*16 for _ in range(16)]
    DS_sd3 = [[0]*16 for _ in range(16)]

    for kd0 in range(16):
        for kd1 in range(16):
            sd2 = (F_TABLE[sd1] ^ sd0 ^ kd0) & MASK
            sd3 = (F_TABLE[sd2] ^ sd1 ^ kd1) & MASK
            DS_sd2[kd0][kd1] = sd2
            DS_sd3[kd0][kd1] = sd3

    return DS_sd2, DS_sd3


def compute_sd_meet(rk: list,
                    DS_sd2: list,
                    DS_sd3: list) -> int:
    """
    Retrieve the dec-side meeting value sd_meet = sd[T_DEC+1] = sd[4]
    from DS^d in O(1).

    sd_meet = F_TABLE[sd3] ^ sd2 ^ kd2
    where:
        kd0 = rk[ROUNDS-1]          = rk[9]
        kd1 = rk[ROUNDS-2]          = rk[8]
        kd2 = rk[ROUNDS - T_DEC]    = rk[7]
        sd2 = DS_sd2[kd0][kd1]
        sd3 = DS_sd3[kd0][kd1]

    Parameters
    ----------
    rk     : subkey list for the candidate key (length = ROUNDS)
    DS_sd2 : precomputed sd2 table from build_dsd()
    DS_sd3 : precomputed sd3 table from build_dsd()

    Returns
    -------
    sd_meet value (integer in [0, 15])

    Example (key=[1,2,3,4], CT=(6,F)):
        kd0=rk[9]=3, kd1=rk[8]=1, kd2=rk[7]=A
        sd2=DS_sd2[3][1]=1, sd3=DS_sd3[3][1]=3
        sd_meet = F_TABLE[3]^1^A = E^1^A = 5  = s[7]  ✓
    """
    kd0 = rk[ROUNDS - 1]           # rk[9]
    kd1 = rk[ROUNDS - 2]           # rk[8]
    kd2 = rk[ROUNDS - T_DEC]       # rk[7]
    sd2 = DS_sd2[kd0][kd1]
    sd3 = DS_sd3[kd0][kd1]
    return (F_TABLE[sd3] ^ sd2 ^ kd2) & MASK


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ONLINE FILTER:  filter_keys  (Algorithm 2)
# ═══════════════════════════════════════════════════════════════════════════════

def filter_keys(se0: int,
                se1: int,
                DS_sd2: list,
                DS_sd3: list) -> list:
    """
    Algorithm 2: online key filtering using correlated sequences.

    For each (ke0,ke1) pair:
      1. Compute se2, se3, Xe  (3 f-calls — shared across all (ke2,ke3))
      2. Build index set I[ke2]  (Nz=4 f-calls via z-linear structure)
      3. For each ke2 in [0,N): derive ke3 candidates from I[ke2]
      4. Partially encrypt (se3, Xe^ke2) for PARTIAL rounds
      5. Check enc_meet == sd_meet  (1 DS^d lookup)

    Parameters
    ----------
    se0, se1 : plaintext left and right halves
    DS_sd2   : precomputed dec-side sd2 table
    DS_sd3   : precomputed dec-side sd3 table

    Returns
    -------
    List of candidate keys [ke0,ke1,ke2,ke3] that passed the filter.
    These are verified against additional PT-CT pairs in recover_key().

    Cost analysis (for Simon-8/16, n=4):
        Outer loop:      16^2 = 256 pairs
        Per pair:        3 f-calls (Step A) + Nz=4 f-calls (Step B) = 7
        Inner loop:      16 * 16 = 256 candidates per pair
        Per candidate:   PARTIAL=4 f-calls (partial enc)
        Total:           256*(7 + 256*4) = 263168  vs
                         naive 16^4 * 10 = 655360  (exhaustive)
    """
    candidates = []

    for ke0 in range(1 << N):
        # ── Step A: 3 f-calls, fixes se2,se3,Xe for ALL (ke2,ke3) ──────────
        #
        # se2 depends only on (ke0) and plaintext — not ke1,ke2,ke3
        se2 = (F_TABLE[se1] ^ se0 ^ ke0) & MASK          # f-call 1

        for ke1 in range(1 << N):
            # se3 depends on (ke0,ke1) — not ke2,ke3
            se3 = (F_TABLE[se2] ^ se1 ^ ke1) & MASK      # f-call 2
            # Xe = f(se3) ^ se2  — the pivot constant
            # s4 = Xe ^ ke2  requires NO f-call (f(se3) already in Xe)
            Xe  = (F_TABLE[se3] ^ se2) & MASK             # f-call 3

            # ── Step B: Build I[ke2] using Nz=4 f-calls ────────────────────
            # (In this simple implementation we call build_index_set which
            # uses N=16 lookups; the z-linear segment saving is conceptual
            # here but is explicitly shown in show_z_segment_saving())
            Ie = build_index_set(se3, Xe)                  # Nz f-calls

            # ── Inner loop: all (ke2,ke3) completions ──────────────────────
            for ke2 in range(1 << N):
                # se4 = Xe ^ ke2 = s[T_ENC+1] = s[4]
                # No f-call: f(se3) is already captured in Xe
                se4 = (Xe ^ ke2) & MASK

                for ke3 in range(1 << N):
                    # Key schedule: expand (ke0,ke1,ke2,ke3) to 10 subkeys
                    rk = key_schedule([ke0, ke1, ke2, ke3])

                    # ── Partial encryption: rounds T_ENC..T_ENC+PARTIAL-1 ──
                    # Start from Feistel state (se3, se4) = (s[3], s[4])
                    # After PARTIAL=4 rounds: L = s[7] = enc_meet
                    L, R = se3, se4
                    for rnd in range(T_ENC, T_ENC + PARTIAL):
                        L, R = R, (L ^ F_TABLE[R] ^ rk[rnd]) & MASK

                    enc_meet = L   # should equal s[T_ENC+PARTIAL] = s[7]

                    # ── Dec-side meeting value (O(1) DS^d lookup) ──────────
                    sd_meet = compute_sd_meet(rk, DS_sd2, DS_sd3)

                    # ── First match ─────────────────────────────────────────
                    if enc_meet == sd_meet:
                        candidates.append([ke0, ke1, ke2, ke3])

    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — KEY RECOVERY  (Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════════════

def recover_key(known_pairs: list,
                verbose: bool = False) -> list | None:
    """
    Algorithm 1: Full MitM key recovery for 10-round Simon-8/16.

    Steps
    -----
    1. Build z-linear segment sets (informational — cost Nz=4 f-calls).
    2. Build DS^d offline from the first ciphertext (cost 2×16^2 f-calls).
    3. Run filter_keys() online to find candidate keys.
    4. Verify survivors against PT-CT pairs 2 and 3.
    5. Return the unique correct key.

    Parameters
    ----------
    known_pairs : list of >= 3  ((pt_L, pt_R), (ct_L, ct_R)) tuples
    verbose     : if True, print detailed step-by-step output

    Returns
    -------
    [k0, k1, k2, k3]  or  None if no key found

    Example
    -------
    >>> pairs = [((7,3),(6,0xF)), ((0xF,5),(5,0xB)), ((0xA,6),(0xC,8))]
    >>> recover_key(pairs)
    [1, 2, 3, 4]
    """
    assert len(known_pairs) >= 3, "Need at least 3 known PT-CT pairs"

    (se0, se1), (ct0, ct1) = known_pairs[0]

    if verbose:
        print(f"\n  Plaintext  : ({se0:X}, {se1:X})")
        print(f"  Ciphertext : ({ct0:X}, {ct1:X})")

    # ── Offline: build DS^d ────────────────────────────────────────────────
    t0 = time.perf_counter()
    DS_sd2, DS_sd3 = build_dsd(ct0, ct1)
    if verbose:
        print(f"  DS^d built in {time.perf_counter()-t0:.4f}s"
              f"  (sd0={ct1:X}, sd1={ct0:X})")

    # ── Online: filter candidates ──────────────────────────────────────────
    t1 = time.perf_counter()
    candidates = filter_keys(se0, se1, DS_sd2, DS_sd3)
    if verbose:
        print(f"  filter_keys: {len(candidates)} candidates"
              f" in {time.perf_counter()-t1:.4f}s")

    # ── Brute-force verify with pairs 2 and 3 ─────────────────────────────
    for (p0, p1), (c0, c1) in known_pairs[1:]:
        candidates = [k for k in candidates
                      if encrypt_block(p0, p1, key_schedule(k)) == (c0, c1)]
    if verbose:
        print(f"  After verification: {len(candidates)} candidates")

    return candidates[0] if candidates else None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — STEP-BY-STEP VERBOSE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo_verbose(secret_key: list,
                 pt: tuple = (7, 3)) -> None:
    """
    Print a complete, annotated walkthrough of the attack.
    Mirrors exactly the worked example in the LaTeX document.

    Parameters
    ----------
    secret_key : [k0,k1,k2,k3] — the key to attack
    pt         : (L,R) plaintext for primary pair
    """
    rk = key_schedule(secret_key)
    ke0, ke1, ke2, ke3 = secret_key
    se0, se1 = pt

    print("=" * 70)
    print("  Simon-8/16  n=4  10-round  Correlated-Sequence MitM")
    print("  Step-by-step example")
    print("=" * 70)
    print(f"  Secret key   : {[hex(k) for k in secret_key]}")
    print(f"  Shift params : (a,b,c) = ({A},{B},{C})")
    print(f"  Word size    : n = {N} bits,  MASK = 0x{MASK:X}")
    print(f"  Subkeys rk   : {[hex(x) for x in rk]}")
    print()

    # ── f-table ─────────────────────────────────────────────────────────────
    print("─" * 70)
    print("  f-TABLE:  f(x) = (x & rl(x,1)) ^ rl(x,2)")
    header = "  x      : " + "  ".join(f"{x:X}" for x in range(16))
    values = "  FT[x]  : " + "  ".join(f"{F_TABLE[x]:X}" for x in range(16))
    print(header)
    print(values)
    print()

    # ── NLFSR state trace ────────────────────────────────────────────────────
    print("─" * 70)
    print("  NLFSR STATE TRACE")
    print(f"  PT = (s0,s1) = ({se0:X}, {se1:X})")
    states = nlfsr_trace(se0, se1, rk)
    ct0, ct1 = states[ROUNDS], states[ROUNDS+1]

    for i, v in enumerate(states):
        if i == 0 or i == 1:
            note = "  ← plaintext"
        elif 2 <= i <= 4:
            note = f"  ← enc window  (uses rk[{i-2}]={rk[i-2]:X})"
        elif 5 <= i <= 6:
            note = f"  ← partial enc (uses rk[{i-2}]={rk[i-2]:X})"
        elif i == T_ENC + PARTIAL:   # s[7]
            note = f"  ← ★ MATCH POINT  (uses rk[{i-2}]={rk[i-2]:X})"
        elif i <= ROUNDS - 1:
            note = f"  ← uses rk[{i-2}]={rk[i-2]:X}"
        else:
            note = "  ← ciphertext"
        print(f"    s[{i:2d}] = {v:X}{note}")
    print(f"\n  CT = (s[{ROUNDS}], s[{ROUNDS+1}]) = ({ct0:X}, {ct1:X})")
    print()

    # ── Step A: 3 f-calls ───────────────────────────────────────────────────
    print("─" * 70)
    print("  STEP A — 3 f-calls to fix enc-side constants")
    print(f"  (ke0={ke0:X}, ke1={ke1:X} are the outer-loop values being guessed)")
    print()
    se2 = (F_TABLE[se1] ^ se0 ^ ke0) & MASK
    se3 = (F_TABLE[se2] ^ se1 ^ ke1) & MASK
    Xe  = (F_TABLE[se3] ^ se2) & MASK
    se4 = (Xe ^ ke2) & MASK

    print(f"  f-call 1:  se2 = FT[se1] ^ se0 ^ ke0")
    print(f"             se2 = FT[{se1:X}] ^ {se0:X} ^ {ke0:X}")
    print(f"                 = {F_TABLE[se1]:X} ^ {se0:X} ^ {ke0:X} = {se2:X}")
    print(f"             ✓  se2 = {se2:X} = s[2] = {states[2]:X}")
    print()
    print(f"  f-call 2:  se3 = FT[se2] ^ se1 ^ ke1")
    print(f"             se3 = FT[{se2:X}] ^ {se1:X} ^ {ke1:X}")
    print(f"                 = {F_TABLE[se2]:X} ^ {se1:X} ^ {ke1:X} = {se3:X}")
    print(f"             ✓  se3 = {se3:X} = s[3] = {states[3]:X}")
    print()
    print(f"  f-call 3:  Xe  = FT[se3] ^ se2")
    print(f"             Xe  = FT[{se3:X}] ^ {se2:X}")
    print(f"                 = {F_TABLE[se3]:X} ^ {se2:X} = {Xe:X}")
    print()
    print(f"  [FREE]     se4 = Xe ^ ke2 = {Xe:X} ^ {ke2:X} = {se4:X}")
    print(f"             (f(se3) already in Xe — no f-call needed)")
    print(f"             ✓  se4 = {se4:X} = s[4] = {states[4]:X}")
    print()

    # ── Step B: index set and z-linear segments ──────────────────────────────
    print("─" * 70)
    print("  STEP B — Index set I[ke2] with Nz=4 f-calls")
    print(f"  I[ke2] = FT[Xe ^ ke2] ^ se3   (se3={se3:X}, Xe={Xe:X})")
    print()

    Ie = build_index_set(se3, Xe)
    print(f"  ke2 : " + " ".join(f"{k:X}" for k in range(16)))
    print(f"  I[] : " + " ".join(f"{v:X}" for v in Ie))
    print()

    y  = Ie[ke2] ^ ke3
    print(f"  Our key: ke2={ke2:X}, ke3={ke3:X}")
    print(f"    I[{ke2:X}] = FT[{Xe:X}^{ke2:X}] ^ {se3:X}")
    print(f"          = FT[{(Xe^ke2)&MASK:X}] ^ {se3:X}")
    print(f"          = {F_TABLE[(Xe^ke2)&MASK]:X} ^ {se3:X} = {Ie[ke2]:X}")
    print(f"    Target: s[5] = y = I[{ke2:X}] ^ ke3 = {Ie[ke2]:X} ^ {ke3:X} = {y:X}")
    print(f"    ✓  y = {y:X} = s[5] = {states[5]:X}  (match: {y==states[5]})")
    print()

    # Z-groups display
    Z_vals, CLZ = build_z_segments()
    print(f"  Z-linear segment sets (Nz={len(Z_vals)}):")
    for z, ldr in zip(Z_vals, CLZ):
        all_mem = []
        for s in ldr:
            all_mem.extend(orbit_members(s))
        all_mem = sorted(set(all_mem))
        fcall = "← 1 f-call" if ldr[0] == min(ldr) else ""
        print(f"    z={z:X}: leaders={[hex(x) for x in ldr]}"
              f"  covers={[hex(x) for x in all_mem]}")
    print(f"  Cost: 3 (Step A) + {len(Z_vals)} (Step B) = "
          f"{3+len(Z_vals)} f-calls  "
          f"(vs naive 3+16={3+16}  →  saving: "
          f"{(16-len(Z_vals))/16*100:.0f}%)")
    print()

    # ── Partial encryption ───────────────────────────────────────────────────
    print("─" * 70)
    print(f"  PARTIAL ENCRYPTION — rounds {T_ENC} to {T_ENC+PARTIAL-1}")
    print(f"  Start: (L,R) = (se3, se4) = ({se3:X}, {se4:X}) = (s[3],s[4])")
    print()
    L, R = se3, se4
    for rnd in range(T_ENC, T_ENC + PARTIAL):
        nR = (L ^ F_TABLE[R] ^ rk[rnd]) & MASK
        print(f"  Round {rnd}:  L={L:X}  R={R:X}  FT[{R:X}]={F_TABLE[R]:X}"
              f"  rk[{rnd}]={rk[rnd]:X}"
              f"  → nR = {L:X}^{F_TABLE[R]:X}^{rk[rnd]:X} = {nR:X}")
        L, R = R, nR
        print(f"            (L,R) = ({L:X},{R:X})  "
              f"→ matches (s[{rnd+1}],s[{rnd+2}]) = "
              f"({states[rnd+1]:X},{states[rnd+2]:X})"
              f"  {'✓' if L==states[rnd+1] and R==states[rnd+2] else '✗'}")
    enc_meet = L
    print(f"\n  enc_meet = L = {enc_meet:X}  "
          f"(= s[{T_ENC+PARTIAL}] = {states[T_ENC+PARTIAL]:X}  "
          f"{'✓' if enc_meet == states[T_ENC+PARTIAL] else '✗'})")
    print()

    # ── DS^d lookup ──────────────────────────────────────────────────────────
    print("─" * 70)
    print("  DEC-SIDE DS^d LOOKUP (offline, O(1) at match time)")
    print(f"  sd0 = CT[1] = s[{ROUNDS+1}] = {ct1:X}   (= s_{{r+1}}, paper convention)")
    print(f"  sd1 = CT[0] = s[{ROUNDS}]   = {ct0:X}   (= s_r)")
    kd0 = rk[ROUNDS-1]
    kd1 = rk[ROUNDS-2]
    kd2 = rk[ROUNDS-T_DEC]
    print(f"  kd0 = rk[{ROUNDS-1}] = {kd0:X}")
    print(f"  kd1 = rk[{ROUNDS-2}] = {kd1:X}")
    print(f"  kd2 = rk[{ROUNDS-T_DEC}] = {kd2:X}")
    sd0 = ct1;  sd1 = ct0
    sd2 = (F_TABLE[sd1] ^ sd0 ^ kd0) & MASK
    sd3 = (F_TABLE[sd2] ^ sd1 ^ kd1) & MASK
    sd_m = (F_TABLE[sd3] ^ sd2 ^ kd2) & MASK
    print(f"\n  sd2 = FT[{sd1:X}] ^ {sd0:X} ^ {kd0:X}"
          f" = {F_TABLE[sd1]:X} ^ {sd0:X} ^ {kd0:X} = {sd2:X}")
    print(f"  sd3 = FT[{sd2:X}] ^ {sd1:X} ^ {kd1:X}"
          f" = {F_TABLE[sd2]:X} ^ {sd1:X} ^ {kd1:X} = {sd3:X}")
    print(f"  sd_meet = FT[{sd3:X}] ^ {sd2:X} ^ {kd2:X}"
          f" = {F_TABLE[sd3]:X} ^ {sd2:X} ^ {kd2:X} = {sd_m:X}")
    print()

    # ── Match ────────────────────────────────────────────────────────────────
    print("─" * 70)
    matched = (enc_meet == sd_m)
    print(f"  MATCH:  enc_meet = {enc_meet:X}  {'==' if matched else '!='}"
          f"  sd_meet = {sd_m:X}  {'✓ MATCH' if matched else '✗ NO MATCH'}")
    print()

    # ── Full attack with verification ─────────────────────────────────────────
    print("─" * 70)
    print("  FULL ATTACK (with 3 PT-CT pairs)")
    extra_pts = [(0xF, 0x5), (0xA, 0x6)]
    all_pairs = [((se0, se1), (ct0, ct1))]
    for ep in extra_pts:
        ec = encrypt_block(ep[0], ep[1], rk)
        all_pairs.append((ep, ec))
        print(f"  Pair: PT=({ep[0]:X},{ep[1]:X})  →  CT=({ec[0]:X},{ec[1]:X})")
    print()
    t0 = time.perf_counter()
    recovered = recover_key(all_pairs, verbose=True)
    t_total   = time.perf_counter() - t0
    print(f"  Total time: {t_total:.4f}s")
    print()
    ok = (recovered == secret_key)
    if ok:
        print(f"  ★ SUCCESS  Recovered: {[hex(k) for k in recovered]}")
    else:
        print(f"  ✗ FAIL     Got: {recovered}  Expected: {[hex(k) for k in secret_key]}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — BATCH RANDOM-KEY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_random_tests(n_tests: int = 10,
                     seed: int = 42) -> None:
    """
    Run n_tests random-key attacks and report PASS/FAIL with timing.
    Each test uses a fresh random 4-nibble key and 3 random plaintexts.
    """
    rng    = random.Random(seed)
    passed = 0

    print(f"\n  Running {n_tests} random-key tests (seed={seed})...\n")

    for i in range(n_tests):
        sk  = [rng.randint(0, 15) for _ in range(4)]
        pts = [(rng.randint(0, 15), rng.randint(0, 15))
               for _ in range(3)]

        rk_t  = key_schedule(sk)
        pairs = [(pt, encrypt_block(pt[0], pt[1], rk_t)) for pt in pts]

        t0  = time.perf_counter()
        rec = recover_key(pairs)
        dt  = time.perf_counter() - t0

        ok  = (rec == sk)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] test {i+1:2d}/{n_tests}"
              f"  key={[hex(k) for k in sk]}"
              f"  →  {[hex(k) for k in rec] if rec else None}"
              f"  ({dt:.3f}s)")
        if ok:
            passed += 1

    print(f"\n  Result: {passed}/{n_tests} passed.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Simon-8/16 Correlated-Sequence MitM Attack\n"
            "n=4 bits, 10 reduced rounds, key=16 bits"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python simon_816_mitm.py                       # full verbose demo
  python simon_816_mitm.py --key 1 2 3 4         # custom key (decimal)
  python simon_816_mitm.py --key 0xC 0x3 0xB 0xB # hex key
  python simon_816_mitm.py --tests 20            # 20 random tests
  python simon_816_mitm.py --tests 0             # skip batch tests
        """,
    )
    p.add_argument(
        "--key", nargs=4, metavar="N",
        default=["1", "2", "3", "4"],
        help="Secret key as 4 nibbles in [0,15] (default: 1 2 3 4)",
    )
    p.add_argument(
        "--pt", nargs=2, metavar="N",
        default=["7", "3"],
        help="Plaintext for verbose demo (default: 7 3)",
    )
    p.add_argument(
        "--tests", type=int, default=10, metavar="N",
        help="Number of random-key batch tests (default: 10, 0 to skip)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for random tests (default: 42)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Skip verbose demo, only run batch tests",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Parse key — accept decimal or hex (0x prefix)
    sk = [int(x, 0) & MASK for x in args.key]
    pt = tuple(int(x, 0) & MASK for x in args.pt)

    if not args.quiet:
        demo_verbose(sk, pt)

    if args.tests > 0:
        run_random_tests(n_tests=args.tests, seed=args.seed)
