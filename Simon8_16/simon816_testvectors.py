"""
Simon-8/16 Test Vector Generator
=================================
Cipher : Simon-8/16
         n=4 bits per word, 8-bit block (L||R), 16-bit key (k0||k1||k2||k3)
         Shift params (a,b,c) = (0,1,2)
         f(x) = (x & rl(x,1)) ^ rl(x,2)
         Full cipher = 32 rounds; any reduced round count accepted.

Usage examples
--------------
  python simon816_testvectors.py
      --> interactive menu

  python simon816_testvectors.py --rounds 10 --pairs 4 --keys 3
      --> 3 random keys x 4 PT-CT pairs x 10 rounds, full printout

  python simon816_testvectors.py --key 1 2 3 4 --pt 7 3 --rounds 10
      --> single key, single PT, verbose trace

  python simon816_testvectors.py --all-rounds --key 1 2 3 4 --pt 7 3
      --> same key/PT, all round counts from 1 to 32

  python simon816_testvectors.py --rounds 10 --pairs 6 --random-keys 5 --seed 99
      --> 5 random keys, 6 random PTs each, non-interactive

  python simon816_testvectors.py --rounds 10 16 32 --pairs 4 --key 1 2 3 4
      --> multiple round counts in one run
"""

import argparse
import random
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CIPHER CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

N    = 4          # word size in bits
MASK = 0xF        # 2^N - 1 = 15

# Shift parameters for Simon-8/16
A, B, C = 0, 1, 2

# Key-schedule LFSR constant z0 (period 62)
Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,
      1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,
      1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]


# ─────────────────────────────────────────────────────────────────────────────
# CORE PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def rl(x: int, r: int) -> int:
    """Cyclic left-rotate an n=4 bit word by r positions."""
    r &= (N - 1)
    return ((x << r) | (x >> (N - r))) & MASK


def f(x: int) -> int:
    """Simon nonlinear function  f(x) = (x & rl(x,1)) ^ rl(x,2)."""
    return (rl(x, A) & rl(x, B)) ^ rl(x, C)


# Pre-computed f-table  FT[x] = f(x)  for x in 0..15
FT = [f(x) for x in range(1 << N)]


def key_schedule(k: list, rounds: int) -> list:
    """
    Expand 4-nibble master key [k0,k1,k2,k3] to `rounds` subkeys.
    CON = 2^n - 4 = 0xC for n=4.
    """
    CON = (MASK - 3) & MASK      # 0xC
    rk  = list(k)
    for i in range(rounds - 4):
        rr3 = ((rk[i+3] >> 3) | (rk[i+3] << (N-3))) & MASK
        rr4 = ((rk[i+3] >> 4) | (rk[i+3] << (N-4))) & MASK
        rr1 = ((rk[i+1] >> 1) | (rk[i+1] << (N-1))) & MASK
        t   = rr3 ^ rr4 ^ rk[i+1] ^ rr1
        rk.append((CON ^ Z0[i % 62] ^ rk[i] ^ t) & MASK)
    return rk[:rounds]


def encrypt(s0: int, s1: int, rk: list) -> tuple:
    """
    Encrypt (s0, s1) for len(rk) rounds.
    Returns (L, R) after all rounds.
    """
    L, R = s0, s1
    for i in range(len(rk)):
        L, R = R, (L ^ FT[R] ^ rk[i]) & MASK
    return L, R


def decrypt(c0: int, c1: int, rk: list) -> tuple:
    """Decrypt (c0, c1) by reversing the Feistel steps."""
    L, R = c0, c1
    for i in reversed(range(len(rk))):
        L, R = (FT[L] ^ R ^ rk[i]) & MASK, L
    return L, R


def nlfsr_states(s0: int, s1: int, rk: list) -> list:
    """Return full NLFSR sequence [s0, s1, s2, ..., s_{rounds+1}]."""
    s = [s0, s1]
    for i in range(len(rk)):
        s.append((FT[s[-1]] ^ s[-2] ^ rk[i]) & MASK)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

SEP_THICK = "=" * 70
SEP_THIN  = "-" * 70


def print_ftable():
    """Pretty-print the complete f-table."""
    print()
    print("  f-table:  f(x) = (x & rl(x,1)) ^ rl(x,2)")
    print("  " + "─" * 56)
    hdr  = "  x    : " + "  ".join(f"{x:X}" for x in range(16))
    vals = "  FT[x]: " + "  ".join(f"{FT[x]:X}" for x in range(16))
    print(hdr)
    print(vals)
    print()


def print_key_schedule(key: list, rounds: int):
    """Print the full subkey expansion for a given key and round count."""
    rk = key_schedule(key, rounds)
    print(f"  Key schedule  key={fmt_key(key)}  ({rounds} rounds)")
    print("  " + "─" * 56)
    per_row = 8
    for row_start in range(0, rounds, per_row):
        indices = range(row_start, min(row_start + per_row, rounds))
        idx_str = "  " + "  ".join(f"rk[{i:2d}]" for i in indices)
        val_str = "  " + "  ".join(f"  {rk[i]:X}  " for i in indices)
        print(idx_str)
        print(val_str)
    print()


def print_nlfsr_trace(s0: int, s1: int, rk: list):
    """
    Print the complete NLFSR state sequence with round annotations.
    """
    r = len(rk)
    s = nlfsr_states(s0, s1, rk)
    print(f"  NLFSR trace  PT=({s0:X},{s1:X})  rounds={r}")
    print("  " + "─" * 56)
    print(f"    s[ 0] = {s[0]:X}  ← plaintext L")
    print(f"    s[ 1] = {s[1]:X}  ← plaintext R")
    for i in range(2, r + 2):
        if i <= r + 1:
            rk_idx = i - 2
            if rk_idx < r:
                note = f"rk[{rk_idx}]={rk[rk_idx]:X}"
            else:
                note = "ciphertext"
            label = "ciphertext" if i >= r + 1 else f"s[{i:2d}] = {s[i]:X}  ← {note}"
            if i > r:
                print(f"    s[{i:2d}] = {s[i]:X}  ← ciphertext {'R' if i==r+1 else 'L'}")
            else:
                print(f"    s[{i:2d}] = {s[i]:X}  ← uses {note}")
    ct = (s[r], s[r+1])
    print(f"\n    CT = (s[{r}],s[{r+1}]) = ({ct[0]:X},{ct[1]:X})")
    # Verify round-by-round
    print("\n  Round-by-round verification:")
    print(f"  {'Rnd':>3}  {'L_in':>5}  {'R_in':>5}  "
          f"{'FT[R]':>6}  {'rk[i]':>6}  "
          f"{'nR=L^FT[R]^rk':>16}  {'(L,R)_out':>12}")
    print("  " + "─" * 70)
    L, R = s0, s1
    for i in range(r):
        nR = (L ^ FT[R] ^ rk[i]) & MASK
        out_L, out_R = R, nR
        print(f"  {i:>3}   "
              f"{L:X}       "
              f"{R:X}       "
              f"{FT[R]:X}       "
              f"{rk[i]:X}       "
              f"{L:X}^{FT[R]:X}^{rk[i]:X}={nR:X}       "
              f"({out_L:X},{out_R:X})")
        L, R = out_L, out_R
    print()


def fmt_key(key: list) -> str:
    return "[" + ",".join(f"{k:X}" for k in key) + "]"


def fmt_nibble(x: int) -> str:
    return f"{x:X}"


def verify_decrypt(s0: int, s1: int, ct: tuple, rk: list) -> bool:
    """Decrypt ct and check it equals (s0,s1)."""
    pt_back = decrypt(ct[0], ct[1], rk)
    return pt_back == (s0, s1)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE KEY TEST VECTOR BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_key_block(key: list,
                               rounds_list: list,
                               pt_pairs: list,
                               show_trace: bool = True,
                               show_schedule: bool = True):
    """
    Generate and print all test vectors for one key across multiple
    round counts and plaintext pairs.

    Parameters
    ----------
    key         : [k0,k1,k2,k3] — 4 nibbles
    rounds_list : list of round counts to test
    pt_pairs    : list of (L,R) plaintext pairs
    show_trace  : print NLFSR state trace for each PT-round combo
    show_schedule: print key schedule expansion
    """
    print(SEP_THICK)
    print(f"  Simon-8/16   n=4   Key = {fmt_key(key)}")
    print(SEP_THICK)

    if show_schedule:
        max_rounds = max(rounds_list)
        print_key_schedule(key, max_rounds)

    # For each round count
    for rounds in rounds_list:
        rk = key_schedule(key, rounds)

        print(f"  ┌─ ROUNDS = {rounds} " + "─" * (55 - len(str(rounds))))
        print(f"  │  Subkeys: {[hex(x) for x in rk]}")
        print(f"  │")

        # Summary table header
        print(f"  │  {'PT (L,R)':>10}   {'CT (L,R)':>10}   "
              f"{'Decrypt OK':>10}   {'FT[L]':>6}  {'FT[R]':>6}")
        print(f"  │  {'─'*10}   {'─'*10}   {'─'*10}   {'─'*6}  {'─'*6}")

        for (pt_l, pt_r) in pt_pairs:
            ct = encrypt(pt_l, pt_r, rk)
            ok = verify_decrypt(pt_l, pt_r, ct, rk)
            print(f"  │  ({pt_l:X},{pt_r:X}) → ({pt_l:X},{pt_r:X})"
                  f"     ({ct[0]:X},{ct[1]:X})"
                  f"          {'✓' if ok else '✗'}"
                  f"           {FT[pt_l]:X}     {FT[pt_r]:X}")

        print(f"  └{'─'*57}")
        print()

        # Detailed NLFSR trace for each PT (if requested)
        if show_trace:
            for (pt_l, pt_r) in pt_pairs:
                print_nlfsr_trace(pt_l, pt_r, rk)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-KEY SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_table(keys_and_pts: list,
                            rounds: int):
    """
    Print a compact summary table for multiple keys and plaintexts
    at a fixed round count.

    keys_and_pts : list of (key, pt_list) where key=[k0,k1,k2,k3]
    """
    print(SEP_THICK)
    print(f"  SUMMARY TABLE   Simon-8/16   {rounds} rounds")
    print(SEP_THICK)
    print(f"  {'Key':>16}  {'PT (L,R)':>10}  {'CT (L,R)':>10}  "
          f"{'rk[0..3]':>20}  {'Dec ✓':>6}")
    print(f"  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*20}  {'─'*6}")

    for key, pts in keys_and_pts:
        rk = key_schedule(key, rounds)
        rk_str = "[" + ",".join(f"{rk[i]:X}" for i in range(min(4,rounds))) + "]"
        for j, (pt_l, pt_r) in enumerate(pts):
            ct  = encrypt(pt_l, pt_r, rk)
            ok  = verify_decrypt(pt_l, pt_r, ct, rk)
            key_str = fmt_key(key) if j == 0 else " " * len(fmt_key(key))
            rk_disp = rk_str        if j == 0 else " " * len(rk_str)
            print(f"  {key_str:>16}  ({pt_l:X},{pt_r:X})      "
                  f"({ct[0]:X},{ct[1]:X})         "
                  f"{rk_disp:>20}  {'✓' if ok else '✗'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ALL-ROUNDS TABLE (one key, one PT, every round from 1..32)
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_rounds_table(key: list, pt: tuple):
    """
    Encrypt `pt` under `key` for every round count from 1 to 32
    and print as a table.
    """
    pt_l, pt_r = pt
    print(SEP_THICK)
    print(f"  ALL-ROUNDS TABLE   Simon-8/16")
    print(f"  Key = {fmt_key(key)}   PT = ({pt_l:X},{pt_r:X})")
    print(SEP_THICK)
    print(f"  {'Rounds':>7}  {'CT (L,R)':>12}  "
          f"{'Last subkey rk[r-1]':>22}  {'Decrypt ✓':>10}")
    print(f"  {'─'*7}  {'─'*12}  {'─'*22}  {'─'*10}")

    for r in range(1, 33):
        rk = key_schedule(key, r)
        ct = encrypt(pt_l, pt_r, rk)
        ok = verify_decrypt(pt_l, pt_r, ct, rk)
        print(f"  {r:>7}  ({ct[0]:X},{ct[1]:X})            "
              f"rk[{r-1:2d}] = {rk[r-1]:X}                    "
              f"{'✓' if ok else '✗'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────────────────────

def get_nibble_list(prompt: str, count: int) -> list:
    """Prompt for `count` nibble values, accept decimal or hex (0x prefix)."""
    while True:
        raw = input(prompt).split()
        if len(raw) == count:
            try:
                vals = [int(x, 0) & MASK for x in raw]
                return vals
            except ValueError:
                pass
        print(f"  ✗  Please enter exactly {count} values (0-15 or 0x0-0xF).")


def get_int(prompt: str, lo: int, hi: int, default: int = None) -> int:
    """Prompt for an integer in [lo, hi]."""
    hint = f" [{lo}-{hi}]" + (f" (default {default})" if default is not None else "")
    while True:
        raw = input(prompt + hint + ": ").strip()
        if raw == "" and default is not None:
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  ✗  Enter an integer between {lo} and {hi}.")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    hint = " [Y/n]" if default else " [y/N]"
    raw = input(prompt + hint + ": ").strip().lower()
    if raw == "":
        return default
    return raw.startswith("y")


def interactive_menu():
    """Full interactive session for generating Simon-8/16 test vectors."""
    print()
    print(SEP_THICK)
    print("  Simon-8/16 Test Vector Generator — Interactive Mode")
    print(f"  Cipher: Simon-8/16  n=4  (a,b,c)=(0,1,2)  f(x)=(x&rl(x,1))^rl(x,2)")
    print(SEP_THICK)

    # ── Show f-table ──────────────────────────────────────────────────────
    print_ftable()

    # ── Mode selection ────────────────────────────────────────────────────
    print("  Mode options:")
    print("    1. Single key  — you specify the key and plaintexts")
    print("    2. Random keys — generator picks random keys and plaintexts")
    print("    3. All-rounds  — one key/PT across every round count (1-32)")
    print()
    mode = get_int("  Choose mode", 1, 3, 1)
    print()

    # ── Round counts ──────────────────────────────────────────────────────
    if mode != 3:
        multi = get_yes_no("  Test multiple round counts?", False)
        if multi:
            raw = input("  Enter round counts (space-separated, each 1-32): ")
            rounds_list = sorted(set(
                max(1, min(32, int(x)))
                for x in raw.split() if x.isdigit()
            ))
            if not rounds_list:
                rounds_list = [10]
        else:
            r = get_int("  Number of rounds", 1, 32, 10)
            rounds_list = [r]
    else:
        rounds_list = list(range(1, 33))

    # ── Number of PT-CT pairs ─────────────────────────────────────────────
    if mode != 3:
        n_pairs = get_int("  Number of PT-CT pairs per key", 1, 16, 4)
    else:
        n_pairs = 1

    # ── Key and PT input ──────────────────────────────────────────────────
    show_trace    = True
    show_schedule = True
    keys_and_pts  = []

    if mode == 1:
        # Manual key
        print()
        n_keys = get_int("  How many keys to test", 1, 10, 1)
        for ki in range(n_keys):
            print(f"\n  ── Key {ki+1} ──")
            key = get_nibble_list(
                f"  Enter key [k0 k1 k2 k3] (4 nibbles 0-F): ", 4
            )
            pts = []
            manual_pt = get_yes_no(
                f"  Enter plaintexts manually (else random)?", True
            )
            if manual_pt:
                for pi in range(n_pairs):
                    pt = get_nibble_list(
                        f"  PT {pi+1}: enter [L R] (2 nibbles 0-F): ", 2
                    )
                    pts.append(tuple(pt))
            else:
                seed = get_int("  Random seed for PTs", 0, 9999, 42)
                rng  = random.Random(seed + ki)
                pts  = [(rng.randint(0,15), rng.randint(0,15))
                        for _ in range(n_pairs)]
            keys_and_pts.append((key, pts))

        show_trace    = get_yes_no("  Show NLFSR state trace for each vector?", True)
        show_schedule = get_yes_no("  Show full key schedule expansion?", True)

    elif mode == 2:
        # Random keys
        n_keys = get_int("  How many random keys", 1, 20, 3)
        seed   = get_int("  Random seed", 0, 9999, 42)
        rng    = random.Random(seed)
        for ki in range(n_keys):
            key = [rng.randint(0,15) for _ in range(4)]
            pts = [(rng.randint(0,15), rng.randint(0,15))
                   for _ in range(n_pairs)]
            keys_and_pts.append((key, pts))
        show_trace    = get_yes_no("  Show NLFSR state trace?", False)
        show_schedule = get_yes_no("  Show key schedule?", False)

    else:
        # All-rounds mode
        key = get_nibble_list(
            "  Enter key [k0 k1 k2 k3] (4 nibbles 0-F): ", 4
        )
        pt  = tuple(get_nibble_list(
            "  Enter plaintext [L R] (2 nibbles 0-F): ", 2
        ))
        keys_and_pts = [(key, [pt])]
        show_trace    = False
        show_schedule = True

    # ── Generate output ───────────────────────────────────────────────────
    print()
    print(SEP_THICK)
    print("  GENERATED TEST VECTORS")
    print(SEP_THICK)

    if mode == 3:
        key, pts = keys_and_pts[0]
        if show_schedule:
            print_key_schedule(key, 32)
        generate_all_rounds_table(key, pts[0])
    else:
        # Detailed per-key blocks
        for key, pts in keys_and_pts:
            generate_single_key_block(
                key, rounds_list, pts,
                show_trace=show_trace,
                show_schedule=show_schedule
            )

        # Compact summary table (if multiple keys or rounds)
        if len(keys_and_pts) > 1 or len(rounds_list) > 1:
            for rounds in rounds_list:
                generate_summary_table(keys_and_pts, rounds)


# ─────────────────────────────────────────────────────────────────────────────
# NON-INTERACTIVE (CLI) MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_cli(args):
    """Non-interactive execution driven by command-line arguments."""

    print()
    print(SEP_THICK)
    print("  Simon-8/16 Test Vector Generator")
    print(SEP_THICK)
    print_ftable()

    # ── Determine round counts ─────────────────────────────────────────────
    if args.all_rounds:
        rounds_list = list(range(1, 33))
    else:
        rounds_list = sorted(set(args.rounds))

    # ── Determine keys and plaintexts ──────────────────────────────────────
    keys_and_pts = []
    rng = random.Random(args.seed)

    if args.key is not None:
        # Single explicit key
        key = [int(x, 0) & MASK for x in args.key]
        if args.pt is not None:
            pts = [tuple(int(x, 0) & MASK for x in args.pt)]
            # top up with randoms if --pairs > 1
            while len(pts) < args.pairs:
                pts.append((rng.randint(0,15), rng.randint(0,15)))
        else:
            pts = [(rng.randint(0,15), rng.randint(0,15))
                   for _ in range(args.pairs)]
        keys_and_pts.append((key, pts))

    if args.random_keys > 0:
        for _ in range(args.random_keys):
            key = [rng.randint(0,15) for _ in range(4)]
            pts = [(rng.randint(0,15), rng.randint(0,15))
                   for _ in range(args.pairs)]
            keys_and_pts.append((key, pts))
    '''
    # Default if nothing specified
    if not keys_and_pts:
        key = [1, 2, 3, 4]
        pts = [(7, 3), (0xF, 5), (0xA, 6), (3, 0xC)][:args.pairs]
        keys_and_pts.append((key, pts))
    '''
    # Default (set by user-1 ) if nothing specified
    if not keys_and_pts:
        key = [10, 11, 12, 13]
        pts = [(15, 15), (0xF, 5), (0xA, 6), (3, 0xC)][:args.pairs]
        keys_and_pts.append((key, pts))

    # ── Output ──────────────────────────────────────────────────────────────
    if args.all_rounds:
        for key, pts in keys_and_pts:
            if args.show_schedule:
                print_key_schedule(key, 32)
            generate_all_rounds_table(key, pts[0])
    else:
        for key, pts in keys_and_pts:
            generate_single_key_block(
                key, rounds_list, pts,
                show_trace=args.trace,
                show_schedule=args.show_schedule
            )
        if len(keys_and_pts) > 1:
            for rounds in rounds_list:
                generate_summary_table(keys_and_pts, rounds)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Simon-8/16 test vector generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  Interactive (menu-driven):
    python simon816_testvectors.py

  Fixed key, fixed PT, verbose trace, rounds 10:
    python simon816_testvectors.py --key 1 2 3 4 --pt 7 3 --rounds 10

  Fixed key, 4 random PTs, rounds 10 and 32:
    python simon816_testvectors.py --key 1 2 3 4 --pairs 4 --rounds 10 32

  All round counts (1-32) for one key+PT:
    python simon816_testvectors.py --key 1 2 3 4 --pt 7 3 --all-rounds

  5 random keys x 6 random PTs x 10 rounds:
    python simon816_testvectors.py --random-keys 5 --pairs 6 --rounds 10

  No NLFSR trace (compact output):
    python simon816_testvectors.py --key 1 2 3 4 --pairs 4 --rounds 10 --no-trace
        """,
    )
    p.add_argument("--key",         nargs=4, metavar="N",
                   help="Master key: 4 nibbles (decimal or 0x hex)")
    p.add_argument("--pt",          nargs=2, metavar="N",
                   help="Plaintext: 2 nibbles")
    p.add_argument("--rounds",      nargs="+", type=int, default=[10],
                   metavar="R",
                   help="Round count(s) to generate (default: 10)")
    p.add_argument("--all-rounds",  action="store_true",
                   help="Generate vectors for every round count 1..32")
    p.add_argument("--pairs",       type=int, default=4, metavar="N",
                   help="Number of PT-CT pairs per key (default: 4)")
    p.add_argument("--random-keys", type=int, default=0, metavar="N",
                   help="Add N additional random keys")
    p.add_argument("--seed",        type=int, default=42,
                   help="RNG seed for random keys/PTs (default: 42)")
    p.add_argument("--trace",       action="store_true", default=True,
                   help="Print NLFSR state trace (default: on)")
    p.add_argument("--no-trace",    dest="trace", action="store_false",
                   help="Suppress NLFSR state trace")
    p.add_argument("--show-schedule", action="store_true", default=True,
                   help="Print key schedule (default: on)")
    p.add_argument("--no-schedule", dest="show_schedule",
                   action="store_false",
                   help="Suppress key schedule printout")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="Force interactive menu even if other flags given")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Use interactive mode when run with no arguments, or when --interactive
    no_args = (len(sys.argv) == 1)
    if no_args or args.interactive:
        interactive_menu()
    else:
        run_cli(args)
