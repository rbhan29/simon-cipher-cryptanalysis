import argparse
import random
import time
from dataclasses import dataclass

import numpy as np

MASK16 = 0xFFFF
FULL_SIDE = 1 << 32


def rol16(x: int, r: int) -> int:
    r &= 15
    return ((x << r) | (x >> (16 - r))) & MASK16


def simon_f(x: int) -> int:
    return (rol16(x, 8) & rol16(x, 1)) ^ rol16(x, 2)


def enc_round(state, k: int):
    l, r = state
    return r, (l ^ simon_f(r) ^ k) & MASK16


def dec_round(state, k: int):
    l1, r1 = state
    l0 = (r1 ^ simon_f(l1) ^ k) & MASK16
    r0 = l1
    return l0, r0


def encrypt_4r(pt, key_words):
    l, r = pt
    for k in key_words:
        l, r = enc_round((l, r), k)
    return l, r


@dataclass
class AttackResult:
    recovered_key: tuple | None
    forward_table_entries: int
    forward_unique_mid_states: int
    forward_build_seconds: float
    backward_scan_seconds: float
    total_seconds: float
    forward_start: int
    forward_stop: int
    backward_start: int
    backward_stop: int
    forward_chunk_index: int
    backward_chunk_index: int


def _chunk_bounds(side_bits: int, chunk_index: int):
    if side_bits < 1 or side_bits > 32:
        raise ValueError("side_bits must be in [1, 32]")
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")

    side = 1 << side_bits
    start = chunk_index * side
    stop = start + side
    if stop > FULL_SIDE:
        raise ValueError(
            f"chunk {chunk_index} with side_bits={side_bits} exceeds 2^32 key space"
        )
    return side, start, stop


def mitm_attack_4r_normal(
    pairs,
    pt_side_bits: int,
    ct_side_bits: int | None = None,
    pt_chunk_index: int = 0,
    ct_chunk_index: int = 0,
) -> AttackResult:
    """
    Normal MITM for 4 rounds with independent round keys:
    - Forward side: enumerate (k0,k1) over PT chunk
    - Backward side: enumerate (k2,k3) over CT chunk
    Meet at state after round 2.
    """
    assert len(pairs) >= 2, "Need at least 2 PT-CT pairs"

    if ct_side_bits is None:
        ct_side_bits = pt_side_bits

    pt_side, pt_start, pt_stop = _chunk_bounds(pt_side_bits, pt_chunk_index)
    ct_side, ct_start, ct_stop = _chunk_bounds(ct_side_bits, ct_chunk_index)

    (pt0, ct0) = pairs[0]

    t0 = time.perf_counter()
    fwd = {}
    for k12 in range(pt_start, pt_stop):
        k0 = k12 & MASK16
        k1 = (k12 >> 16) & MASK16
        s1 = enc_round(pt0, k0)
        s2 = enc_round(s1, k1)
        mid = (s2[0] << 16) | s2[1]
        bucket = fwd.get(mid)
        if bucket is None:
            fwd[mid] = [k12]
        else:
            bucket.append(k12)
    t1 = time.perf_counter()

    found_key = None

    for k34 in range(ct_start, ct_stop):
        k2 = k34 & MASK16
        k3 = (k34 >> 16) & MASK16

        s3 = dec_round(ct0, k3)
        s2 = dec_round(s3, k2)
        mid = (s2[0] << 16) | s2[1]

        cands = fwd.get(mid)
        if not cands:
            continue

        for k12 in cands:
            k0 = k12 & MASK16
            k1 = (k12 >> 16) & MASK16
            key = (k0, k1, k2, k3)
            ok = True
            for pt, ct in pairs[1:]:
                if encrypt_4r(pt, key) != ct:
                    ok = False
                    break
            if ok:
                found_key = key
                break

        if found_key is not None:
            break

    t2 = time.perf_counter()
    return AttackResult(
        recovered_key=found_key,
        forward_table_entries=pt_side,
        forward_unique_mid_states=len(fwd),
        forward_build_seconds=t1 - t0,
        backward_scan_seconds=t2 - t1,
        total_seconds=t2 - t0,
        forward_start=pt_start,
        forward_stop=pt_stop,
        backward_start=ct_start,
        backward_stop=ct_stop,
        forward_chunk_index=pt_chunk_index,
        backward_chunk_index=ct_chunk_index,
    )


def simon_f_vec_u32(x_u32: np.ndarray) -> np.ndarray:
    x = x_u32 & np.uint32(MASK16)
    rl1 = ((x << np.uint32(1)) | (x >> np.uint32(15))) & np.uint32(MASK16)
    rl8 = ((x << np.uint32(8)) | (x >> np.uint32(8))) & np.uint32(MASK16)
    rl2 = ((x << np.uint32(2)) | (x >> np.uint32(14))) & np.uint32(MASK16)
    return (rl8 & rl1) ^ rl2


def benchmark_forward_generation(pt, samples: int, batch: int = 1 << 20):
    p0 = np.uint32(pt[0])
    p1 = np.uint32(pt[1])
    fp1 = np.uint32(simon_f(int(p1)))

    acc = np.uint64(0xDEADBEEFCAFEBABE)
    start = time.perf_counter()

    for base in range(0, samples, batch):
        end = min(base + batch, samples)
        idx = np.arange(base, end, dtype=np.uint32)
        k0 = idx & np.uint32(MASK16)
        k1 = idx >> np.uint32(16)

        r1 = (p0 ^ fp1 ^ k0) & np.uint32(MASK16)
        l2 = r1
        r2 = (p1 ^ simon_f_vec_u32(r1) ^ k1) & np.uint32(MASK16)
        mid = (l2 << np.uint32(16)) | r2
        chunk_hash = (mid.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)).sum(dtype=np.uint64)
        acc = np.uint64(acc ^ chunk_hash ^ np.uint64(base) ^ np.uint64(end - base))

    elapsed = time.perf_counter() - start
    return elapsed, int(acc)


def benchmark_backward_generation(ct, samples: int, batch: int = 1 << 20):
    c0 = np.uint32(ct[0])
    c1 = np.uint32(ct[1])
    fc0 = np.uint32(simon_f(int(c0)))

    acc = np.uint64(0xDEADBEEFCAFEBABE)
    start = time.perf_counter()

    for base in range(0, samples, batch):
        end = min(base + batch, samples)
        idx = np.arange(base, end, dtype=np.uint32)
        k2 = idx & np.uint32(MASK16)
        k3 = idx >> np.uint32(16)

        l3 = (c1 ^ fc0 ^ k3) & np.uint32(MASK16)
        r3 = np.full(end - base, c0, dtype=np.uint32)

        l2 = (r3 ^ simon_f_vec_u32(l3) ^ k2) & np.uint32(MASK16)
        r2 = l3
        mid = (l2 << np.uint32(16)) | r2
        chunk_hash = (mid.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)).sum(dtype=np.uint64)
        acc = np.uint64(acc ^ chunk_hash ^ np.uint64(base) ^ np.uint64(end - base))

    elapsed = time.perf_counter() - start
    return elapsed, int(acc)


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.3f} s"
    mins = sec / 60.0
    if mins < 60:
        return f"{mins:.3f} min"
    hours = mins / 60.0
    if hours < 24:
        return f"{hours:.3f} h"
    days = hours / 24.0
    return f"{days:.3f} d"


def gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def build_pairs_for_demo(
    seed: int,
    pt_side_bits: int,
    ct_side_bits: int,
    n_pairs: int,
    pt_chunk_index: int,
    pt_chunk_count: int,
    ct_chunk_index: int,
):
    rng = random.Random(seed)
    pt_side, pt_start, _ = _chunk_bounds(pt_side_bits, pt_chunk_index)
    ct_side, ct_start, _ = _chunk_bounds(ct_side_bits, ct_chunk_index)

    if pt_chunk_count <= 0:
        raise ValueError("pt_chunk_count must be >= 1")
    max_chunks = 1 << (32 - pt_side_bits)
    if pt_chunk_index + pt_chunk_count > max_chunks:
        raise ValueError("pt_chunk_index + pt_chunk_count exceeds PT chunk space")

    chosen_pt_chunk = pt_chunk_index + rng.randrange(pt_chunk_count)
    chosen_pt_start = chosen_pt_chunk * pt_side

    k12 = chosen_pt_start + rng.randrange(pt_side)
    k34 = ct_start + rng.randrange(ct_side)
    key = (
        k12 & MASK16,
        (k12 >> 16) & MASK16,
        k34 & MASK16,
        (k34 >> 16) & MASK16,
    )

    pairs = []
    for _ in range(n_pairs):
        pt = (rng.randrange(1 << 16), rng.randrange(1 << 16))
        ct = encrypt_4r(pt, key)
        pairs.append((pt, ct))

    return key, pairs, chosen_pt_chunk


def main():
    parser = argparse.ArgumentParser(description="Normal MITM for 4-round SIMON32/64")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--demo-side-bits", type=int, default=20,
                        help="Legacy alias for --pt-side-bits/--ct-side-bits")
    parser.add_argument("--pt-side-bits", type=int, default=None,
                        help="PT-side chunk size: 2^bits candidates for (k0,k1)")
    parser.add_argument("--ct-side-bits", type=int, default=None,
                        help="CT-side chunk size: 2^bits candidates for (k2,k3)")
    parser.add_argument("--pt-chunk-index", type=int, default=0,
                        help="PT chunk index to start from")
    parser.add_argument("--ct-chunk-index", type=int, default=0,
                        help="CT chunk index")
    parser.add_argument("--pt-chunk-count", type=int, default=1,
                        help="Number of sequential PT chunks to scan")
    parser.add_argument("--continue-after-hit", action="store_true",
                        help="Continue scanning remaining PT chunks even after a key is found")
    parser.add_argument("--benchmark-samples", type=int, default=(1 << 22),
                        help="Number of candidates per side used for throughput benchmark")
    parser.add_argument("--report-file", type=str, default="MITM_4R_NORMAL_RESULTS.txt")
    parser.add_argument("--chunk-report-file", type=str, default="MITM_4R_PT_CHUNK_RESULTS.txt",
                        help="Separate report for PT-chunk sweep runs")
    args = parser.parse_args()

    pt_side_bits = args.pt_side_bits if args.pt_side_bits is not None else args.demo_side_bits
    ct_side_bits = args.ct_side_bits if args.ct_side_bits is not None else pt_side_bits

    if pt_side_bits < 1 or pt_side_bits > 32:
        raise ValueError("--pt-side-bits must be in [1, 32]")
    if ct_side_bits < 1 or ct_side_bits > 32:
        raise ValueError("--ct-side-bits must be in [1, 32]")
    if args.pairs < 2:
        raise ValueError("--pairs must be >= 2")
    if args.benchmark_samples <= 0:
        raise ValueError("--benchmark-samples must be positive")
    if args.pt_chunk_count <= 0:
        raise ValueError("--pt-chunk-count must be >= 1")

    max_pt_chunks = 1 << (32 - pt_side_bits)
    if args.pt_chunk_index < 0 or args.pt_chunk_index + args.pt_chunk_count > max_pt_chunks:
        raise ValueError("PT chunk range exceeds 2^32 PT-side space")
    max_ct_chunks = 1 << (32 - ct_side_bits)
    if args.ct_chunk_index < 0 or args.ct_chunk_index >= max_ct_chunks:
        raise ValueError("CT chunk index exceeds 2^32 CT-side space")

    demo_key, pairs, chosen_pt_chunk = build_pairs_for_demo(
        seed=args.seed,
        pt_side_bits=pt_side_bits,
        ct_side_bits=ct_side_bits,
        n_pairs=args.pairs,
        pt_chunk_index=args.pt_chunk_index,
        pt_chunk_count=args.pt_chunk_count,
        ct_chunk_index=args.ct_chunk_index,
    )

    first_result = None
    attempted_chunks = 0
    hit_result = None
    hit_chunk = None

    with open(args.chunk_report_file, "w", encoding="utf-8") as f:
        f.write("SIMON32/64 4-Round Normal MITM PT-Chunk Sweep Report\n")
        f.write("Date: 2026-03-29\n\n")
        f.write("Sweep setup\n")
        f.write(f"- PT-side chunk size: 2^{pt_side_bits}\n")
        f.write(f"- CT-side chunk size: 2^{ct_side_bits}\n")
        f.write(f"- PT chunk start index: {args.pt_chunk_index}\n")
        f.write(f"- PT chunk count requested: {args.pt_chunk_count}\n")
        f.write(f"- CT chunk index: {args.ct_chunk_index}\n")
        f.write(f"- Continue after hit: {args.continue_after_hit}\n")
        f.write(f"- Generated key PT chunk: {chosen_pt_chunk}\n")
        f.write("\nPer-chunk progress\n")
    for off in range(args.pt_chunk_count):
        pt_chunk = args.pt_chunk_index + off
        res = mitm_attack_4r_normal(
            pairs,
            pt_side_bits=pt_side_bits,
            ct_side_bits=ct_side_bits,
            pt_chunk_index=pt_chunk,
            ct_chunk_index=args.ct_chunk_index,
        )
        attempted_chunks += 1
        if first_result is None:
            first_result = res

        with open(args.chunk_report_file, "a", encoding="utf-8") as f:
            f.write(
                f"- PT chunk {res.forward_chunk_index}: "
                f"time={res.total_seconds:.6f}s "
                f"forward={res.forward_build_seconds:.6f}s "
                f"backward={res.backward_scan_seconds:.6f}s "
                f"hit={res.recovered_key is not None}\n"
            )

        if res.recovered_key is not None and hit_result is None:
            hit_result = res
            hit_chunk = pt_chunk
            if not args.continue_after_hit:
                break

    demo = first_result
    if demo is None:
        raise RuntimeError("No PT chunks were scanned")

    bench_f_sec, f_checksum = benchmark_forward_generation(pairs[0][0], args.benchmark_samples)
    bench_b_sec, b_checksum = benchmark_backward_generation(pairs[0][1], args.benchmark_samples)

    f_rate = args.benchmark_samples / bench_f_sec
    b_rate = args.benchmark_samples / bench_b_sec

    scale_from_demo = 2 ** (32 - pt_side_bits)
    proj_forward_from_demo = demo.forward_build_seconds * scale_from_demo
    proj_backward_from_demo = demo.backward_scan_seconds * scale_from_demo
    proj_total_from_demo = demo.total_seconds * scale_from_demo

    est_forward_32 = FULL_SIDE / f_rate
    est_backward_32 = FULL_SIDE / b_rate
    est_total_32 = est_forward_32 + est_backward_32

    packed_forward_bytes = FULL_SIDE * 8
    packed_both_bytes = packed_forward_bytes * 2

    lines = [
        "SIMON32/64 4-Round Normal MITM Report",
        "Date: 2026-03-29",
        "",
        "Setup",
        f"- Demo key (k0,k1,k2,k3): {[hex(x) for x in demo_key]}",
        f"- Number of PT-CT pairs: {args.pairs}",
        f"- PT-side chunk size: 2^{pt_side_bits}",
        f"- CT-side chunk size: 2^{ct_side_bits}",
        f"- PT chunk start index: {args.pt_chunk_index}",
        f"- PT chunk count requested: {args.pt_chunk_count}",
        f"- CT chunk index: {args.ct_chunk_index}",
        f"- Key was generated in PT chunk: {chosen_pt_chunk}",
        f"- Benchmark samples per side: {args.benchmark_samples}",
        "",
        "1) Exact normal MITM chunk run (same algorithm)",
        f"- Forward table entries: {demo.forward_table_entries}",
        f"- Unique meet states: {demo.forward_unique_mid_states}",
        f"- Forward build time: {demo.forward_build_seconds:.6f} s",
        f"- Backward scan time: {demo.backward_scan_seconds:.6f} s",
        f"- Total exact demo time: {demo.total_seconds:.6f} s",
        f"- Recovered key in first scanned chunk: {[hex(x) for x in demo.recovered_key] if demo.recovered_key else None}",
        f"- PT chunk scanned in first run: {demo.forward_chunk_index}",
        f"- Exact recovery success in first run: {demo.recovered_key == demo_key}",
        f"- Projected forward time at 2^32 from exact impl: {format_seconds(proj_forward_from_demo)}",
        f"- Projected backward time at 2^32 from exact impl: {format_seconds(proj_backward_from_demo)}",
        f"- Projected total time at 2^32 from exact impl: {format_seconds(proj_total_from_demo)}",
        "",
        "2) Full 2^32-per-side core-generation estimate (vectorized benchmark)",
        f"- Forward generation benchmark time: {bench_f_sec:.6f} s",
        f"- Backward generation benchmark time: {bench_b_sec:.6f} s",
        f"- Forward throughput: {f_rate:,.2f} candidates/s",
        f"- Backward throughput: {b_rate:,.2f} candidates/s",
        f"- Estimated forward time for 2^32: {format_seconds(est_forward_32)}",
        f"- Estimated backward time for 2^32: {format_seconds(est_backward_32)}",
        f"- Estimated total core MITM time (no sort/hash overhead): {format_seconds(est_total_32)}",
        f"- Forward checksum guard: 0x{f_checksum:016X}",
        f"- Backward checksum guard: 0x{b_checksum:016X}",
        "",
        "3) Space for normal MITM at 2^32 forward entries",
        "- Packed forward table assumption: store (mid_state32, key12_32) per entry",
        f"- Forward table bytes: {packed_forward_bytes:,} ({gib(packed_forward_bytes):.3f} GiB)",
        f"- If both forward+backward tables are stored: {packed_both_bytes:,} ({gib(packed_both_bytes):.3f} GiB)",
        "",
        "Notes",
        "- This is the normal split: 2 rounds from plaintext side, 2 rounds from ciphertext side.",
        "- PT chunking lets you cover the full PT-side key space in fixed-memory slices.",
        "- Full-scale time above is based on measured candidate generation throughput and does not include",
        "  extra overhead from external sorting or hash-table collision handling at 2^32 scale.",
    ]

    with open(args.report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(args.chunk_report_file, "a", encoding="utf-8") as f:
        f.write("\nSummary\n")
        f.write(f"- PT chunks attempted: {attempted_chunks}\n")
        f.write(f"- Found key chunk: {hit_chunk if hit_chunk is not None else 'not found'}\n")
        f.write(f"- Found key value: {[hex(x) for x in hit_result.recovered_key] if hit_result else None}\n")

    print("\n".join(lines))
    print(f"\nReport written to: {args.report_file}")
    print(f"Chunk report written to: {args.chunk_report_file}")


if __name__ == "__main__":
    main()
