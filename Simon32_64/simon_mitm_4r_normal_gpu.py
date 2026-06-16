import argparse
import random
import time
from dataclasses import dataclass

import numpy as np
from numba import cuda

MASK16 = np.uint32(0xFFFF)
FULL_SIDE = 1 << 32


@cuda.jit(device=True, inline=True)
def _rol16_cuda(x, r):
    rr = r & 15
    return ((x << rr) | (x >> (16 - rr))) & np.uint32(0xFFFF)


@cuda.jit(device=True, inline=True)
def _f_cuda(x):
    return (_rol16_cuda(x, 8) & _rol16_cuda(x, 1)) ^ _rol16_cuda(x, 2)


@cuda.jit
def _kernel_forward_mids(pt0, pt1, k12_start, out_mids):
    idx = cuda.grid(1)
    if idx >= out_mids.size:
        return

    k12 = k12_start + np.uint32(idx)
    k0 = k12 & np.uint32(0xFFFF)
    k1 = (k12 >> np.uint32(16)) & np.uint32(0xFFFF)

    r1 = (pt0 ^ _f_cuda(pt1) ^ k0) & np.uint32(0xFFFF)
    l2 = r1
    r2 = (pt1 ^ _f_cuda(r1) ^ k1) & np.uint32(0xFFFF)

    out_mids[idx] = ((l2 << np.uint32(16)) | r2) & np.uint32(0xFFFFFFFF)


@cuda.jit
def _kernel_backward_mids(ct0, ct1, k34_start, out_mids):
    idx = cuda.grid(1)
    if idx >= out_mids.size:
        return

    k34 = k34_start + np.uint32(idx)
    k2 = k34 & np.uint32(0xFFFF)
    k3 = (k34 >> np.uint32(16)) & np.uint32(0xFFFF)

    l4 = ct0
    r4 = ct1

    l3 = (r4 ^ _f_cuda(l4) ^ k3) & np.uint32(0xFFFF)
    r3 = l4

    l2 = (r3 ^ _f_cuda(l3) ^ k2) & np.uint32(0xFFFF)
    r2 = l3

    out_mids[idx] = ((l2 << np.uint32(16)) | r2) & np.uint32(0xFFFFFFFF)


@cuda.jit
def _kernel_verify_keys(
    k0_arr,
    k1_arr,
    k2_arr,
    k3_arr,
    pt_l_arr,
    pt_r_arr,
    ct_l_arr,
    ct_r_arr,
    n_pairs,
    out_valid,
):
    idx = cuda.grid(1)
    if idx >= out_valid.size:
        return

    k0 = np.uint32(k0_arr[idx])
    k1 = np.uint32(k1_arr[idx])
    k2 = np.uint32(k2_arr[idx])
    k3 = np.uint32(k3_arr[idx])

    ok = np.uint8(1)

    for j in range(n_pairs):
        l = np.uint32(pt_l_arr[j])
        r = np.uint32(pt_r_arr[j])

        nr = (l ^ _f_cuda(r) ^ k0) & np.uint32(0xFFFF)
        l = r
        r = nr

        nr = (l ^ _f_cuda(r) ^ k1) & np.uint32(0xFFFF)
        l = r
        r = nr

        nr = (l ^ _f_cuda(r) ^ k2) & np.uint32(0xFFFF)
        l = r
        r = nr

        nr = (l ^ _f_cuda(r) ^ k3) & np.uint32(0xFFFF)
        l = r
        r = nr

        if l != np.uint32(ct_l_arr[j]) or r != np.uint32(ct_r_arr[j]):
            ok = np.uint8(0)
            break

    out_valid[idx] = ok


@dataclass
class ChunkAttackResult:
    pt_chunk_index: int
    ct_chunk_index: int
    forward_seconds: float
    sort_seconds: float
    backward_seconds: float
    match_seconds: float
    total_seconds: float
    matched_backward_candidates: int
    candidate_pairs_checked: int
    recovered_key: tuple | None


def rol16(x: int, r: int) -> int:
    rr = r & 15
    return ((x << rr) | (x >> (16 - rr))) & 0xFFFF


def simon_f(x: int) -> int:
    return (rol16(x, 8) & rol16(x, 1)) ^ rol16(x, 2)


def enc_round(state, k: int):
    l, r = state
    return r, (l ^ simon_f(r) ^ k) & 0xFFFF


def encrypt_4r(pt, key_words):
    l, r = pt
    for k in key_words:
        l, r = enc_round((l, r), k)
    return l, r


def verify_key_on_pairs(key_words, pairs) -> bool:
    for pt, ct in pairs:
        if encrypt_4r(pt, key_words) != ct:
            return False
    return True


def verify_candidate_keys_gpu(candidate_keys, pairs, threads_per_block: int):
    """Return first valid key from candidate_keys using GPU verification, else None."""
    if candidate_keys.size == 0:
        return None

    keys = np.asarray(candidate_keys, dtype=np.uint16)
    n = keys.shape[0]

    k0 = np.ascontiguousarray(keys[:, 0])
    k1 = np.ascontiguousarray(keys[:, 1])
    k2 = np.ascontiguousarray(keys[:, 2])
    k3 = np.ascontiguousarray(keys[:, 3])

    pt_l = np.array([pt[0] for pt, _ in pairs], dtype=np.uint16)
    pt_r = np.array([pt[1] for pt, _ in pairs], dtype=np.uint16)
    ct_l = np.array([ct[0] for _, ct in pairs], dtype=np.uint16)
    ct_r = np.array([ct[1] for _, ct in pairs], dtype=np.uint16)

    d_k0 = cuda.to_device(k0)
    d_k1 = cuda.to_device(k1)
    d_k2 = cuda.to_device(k2)
    d_k3 = cuda.to_device(k3)
    d_pt_l = cuda.to_device(pt_l)
    d_pt_r = cuda.to_device(pt_r)
    d_ct_l = cuda.to_device(ct_l)
    d_ct_r = cuda.to_device(ct_r)
    d_valid = cuda.device_array(n, dtype=np.uint8)

    blocks = (n + threads_per_block - 1) // threads_per_block
    _kernel_verify_keys[blocks, threads_per_block](
        d_k0,
        d_k1,
        d_k2,
        d_k3,
        d_pt_l,
        d_pt_r,
        d_ct_l,
        d_ct_r,
        np.int32(len(pairs)),
        d_valid,
    )
    cuda.synchronize()

    valid = d_valid.copy_to_host()
    good_idx = np.nonzero(valid)[0]
    if good_idx.size == 0:
        return None

    k = keys[int(good_idx[0])]
    return (int(k[0]), int(k[1]), int(k[2]), int(k[3]))


def chunk_bounds(side_bits: int, chunk_index: int):
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
    pt_side, _, _ = chunk_bounds(pt_side_bits, pt_chunk_index)
    ct_side, ct_start, _ = chunk_bounds(ct_side_bits, ct_chunk_index)

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
        k12 & 0xFFFF,
        (k12 >> 16) & 0xFFFF,
        k34 & 0xFFFF,
        (k34 >> 16) & 0xFFFF,
    )

    pairs = []
    for _ in range(n_pairs):
        pt = (rng.randrange(1 << 16), rng.randrange(1 << 16))
        ct = encrypt_4r(pt, key)
        pairs.append((pt, ct))

    return key, pairs, chosen_pt_chunk


def generate_forward_mids_gpu(pt, k12_start: int, side: int, threads_per_block: int):
    out_dev = cuda.device_array(side, dtype=np.uint32)
    blocks = (side + threads_per_block - 1) // threads_per_block
    _kernel_forward_mids[blocks, threads_per_block](
        np.uint32(pt[0]), np.uint32(pt[1]), np.uint32(k12_start), out_dev
    )
    cuda.synchronize()
    return out_dev.copy_to_host()


def generate_backward_mids_gpu(ct, k34_start: int, side: int, threads_per_block: int):
    out_dev = cuda.device_array(side, dtype=np.uint32)
    blocks = (side + threads_per_block - 1) // threads_per_block
    _kernel_backward_mids[blocks, threads_per_block](
        np.uint32(ct[0]), np.uint32(ct[1]), np.uint32(k34_start), out_dev
    )
    cuda.synchronize()
    return out_dev.copy_to_host()


def attack_chunk_gpu(
    pairs,
    pt_side_bits: int,
    ct_side_bits: int,
    pt_chunk_index: int,
    ct_chunk_index: int,
    threads_per_block: int,
) -> ChunkAttackResult:
    pt_side, pt_start, _ = chunk_bounds(pt_side_bits, pt_chunk_index)
    ct_side, ct_start, _ = chunk_bounds(ct_side_bits, ct_chunk_index)

    (pt0, ct0) = pairs[0]

    t0 = time.perf_counter()
    mids_fwd = generate_forward_mids_gpu(pt0, pt_start, pt_side, threads_per_block)
    t1 = time.perf_counter()

    order = np.argsort(mids_fwd, kind="stable")
    mids_sorted = mids_fwd[order]
    mids_unique, first_idx, counts = np.unique(
        mids_sorted, return_index=True, return_counts=True
    )
    t2 = time.perf_counter()

    mids_bwd = generate_backward_mids_gpu(ct0, ct_start, ct_side, threads_per_block)
    t3 = time.perf_counter()

    pos = np.searchsorted(mids_unique, mids_bwd)
    valid = pos < mids_unique.size
    matches = np.zeros_like(valid)
    valid_pos = pos[valid]
    matches[valid] = mids_unique[valid_pos] == mids_bwd[valid]
    matched_bwd_idx = np.nonzero(matches)[0]
    matched_pos = pos[matches]

    if matched_bwd_idx.size == 0:
        t4 = time.perf_counter()
        return ChunkAttackResult(
            pt_chunk_index=pt_chunk_index,
            ct_chunk_index=ct_chunk_index,
            forward_seconds=t1 - t0,
            sort_seconds=t2 - t1,
            backward_seconds=t3 - t2,
            match_seconds=t4 - t3,
            total_seconds=t4 - t0,
            matched_backward_candidates=0,
            candidate_pairs_checked=0,
            recovered_key=None,
        )

    candidate_pairs_checked = 0
    recovered_key = None
    verify_batch = 1 << 18

    matched_counts = counts[matched_pos]
    single_mask = matched_counts == 1

    def verify_in_batches(cand_keys_u16: np.ndarray):
        nonlocal candidate_pairs_checked
        if cand_keys_u16.size == 0:
            return None
        n = cand_keys_u16.shape[0]
        for base in range(0, n, verify_batch):
            end = min(base + verify_batch, n)
            chunk = cand_keys_u16[base:end]
            candidate_pairs_checked += (end - base)
            found = verify_candidate_keys_gpu(chunk, pairs, threads_per_block)
            if found is not None:
                return found
        return None

    # Fast path: for unique meet-states, each backward candidate maps to one forward key.
    if np.any(single_mask):
        single_pos = matched_pos[single_mask]
        single_bidx = matched_bwd_idx[single_mask]

        fidx_single = order[first_idx[single_pos]]
        k12_single = (pt_start + fidx_single.astype(np.uint64)).astype(np.uint64)
        k34_single = (ct_start + single_bidx.astype(np.uint64)).astype(np.uint64)

        n_single = k12_single.size
        cand_single = np.empty((n_single, 4), dtype=np.uint16)
        cand_single[:, 0] = (k12_single & np.uint64(0xFFFF)).astype(np.uint16)
        cand_single[:, 1] = ((k12_single >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16)
        cand_single[:, 2] = (k34_single & np.uint64(0xFFFF)).astype(np.uint16)
        cand_single[:, 3] = ((k34_single >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16)

        recovered_key = verify_in_batches(cand_single)

    # Fallback path: handle collisions where one backward candidate maps to multiple forward keys.
    if recovered_key is None and np.any(~single_mask):
        multi_pos = matched_pos[~single_mask]
        multi_bidx = matched_bwd_idx[~single_mask]

        batch = []
        for p, b_idx in zip(multi_pos.tolist(), multi_bidx.tolist()):
            s = int(first_idx[p])
            e = s + int(counts[p])
            k34 = ct_start + int(b_idx)
            k2 = k34 & 0xFFFF
            k3 = (k34 >> 16) & 0xFFFF

            for f_idx in order[s:e]:
                k12 = pt_start + int(f_idx)
                k0 = k12 & 0xFFFF
                k1 = (k12 >> 16) & 0xFFFF
                batch.append((k0, k1, k2, k3))

                if len(batch) >= verify_batch:
                    cand = np.array(batch, dtype=np.uint16)
                    batch.clear()
                    recovered_key = verify_in_batches(cand)
                    if recovered_key is not None:
                        break

            if recovered_key is not None:
                break

        if recovered_key is None and batch:
            cand = np.array(batch, dtype=np.uint16)
            recovered_key = verify_in_batches(cand)

    t4 = time.perf_counter()
    return ChunkAttackResult(
        pt_chunk_index=pt_chunk_index,
        ct_chunk_index=ct_chunk_index,
        forward_seconds=t1 - t0,
        sort_seconds=t2 - t1,
        backward_seconds=t3 - t2,
        match_seconds=t4 - t3,
        total_seconds=t4 - t0,
        matched_backward_candidates=int(matched_bwd_idx.size),
        candidate_pairs_checked=candidate_pairs_checked,
        recovered_key=recovered_key,
    )


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated normal MITM for 4-round SIMON32/64"
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--pt-side-bits", type=int, default=24)
    parser.add_argument("--ct-side-bits", type=int, default=None)
    parser.add_argument("--pt-chunk-index", type=int, default=0)
    parser.add_argument("--pt-chunk-count", type=int, default=1)
    parser.add_argument("--ct-chunk-index", type=int, default=0)
    parser.add_argument("--continue-after-hit", action="store_true")
    parser.add_argument("--threads-per-block", type=int, default=256)
    parser.add_argument("--report-file", type=str, default="MITM_4R_NORMAL_GPU_RESULTS.txt")
    parser.add_argument(
        "--chunk-report-file",
        type=str,
        default="MITM_4R_PT_CHUNK_GPU_RESULTS.txt",
    )
    args = parser.parse_args()

    if not cuda.is_available():
        raise RuntimeError("Numba CUDA runtime unavailable in this Python environment")

    ct_side_bits = args.ct_side_bits if args.ct_side_bits is not None else args.pt_side_bits

    if args.pt_side_bits < 1 or args.pt_side_bits > 32:
        raise ValueError("--pt-side-bits must be in [1, 32]")
    if ct_side_bits < 1 or ct_side_bits > 32:
        raise ValueError("--ct-side-bits must be in [1, 32]")
    if args.pairs < 2:
        raise ValueError("--pairs must be >= 2")
    if args.pt_chunk_count <= 0:
        raise ValueError("--pt-chunk-count must be >= 1")
    if args.threads_per_block <= 0 or args.threads_per_block > 1024:
        raise ValueError("--threads-per-block must be in [1, 1024]")

    max_pt_chunks = 1 << (32 - args.pt_side_bits)
    if args.pt_chunk_index < 0 or args.pt_chunk_index + args.pt_chunk_count > max_pt_chunks:
        raise ValueError("PT chunk range exceeds 2^32 PT-side space")

    max_ct_chunks = 1 << (32 - ct_side_bits)
    if args.ct_chunk_index < 0 or args.ct_chunk_index >= max_ct_chunks:
        raise ValueError("CT chunk index exceeds 2^32 CT-side space")

    device = cuda.get_current_device()
    device_name_raw = device.name
    if isinstance(device_name_raw, (bytes, bytearray)):
        device_name = device_name_raw.decode("utf-8", errors="replace")
    else:
        device_name = str(device_name_raw)

    demo_key, pairs, chosen_pt_chunk = build_pairs_for_demo(
        seed=args.seed,
        pt_side_bits=args.pt_side_bits,
        ct_side_bits=ct_side_bits,
        n_pairs=args.pairs,
        pt_chunk_index=args.pt_chunk_index,
        pt_chunk_count=args.pt_chunk_count,
        ct_chunk_index=args.ct_chunk_index,
    )

    # Warm up CUDA kernels so first measured chunk excludes JIT compilation overhead.
    _ = generate_forward_mids_gpu(pairs[0][0], 0, 1, args.threads_per_block)
    _ = generate_backward_mids_gpu(pairs[0][1], 0, 1, args.threads_per_block)
    _ = verify_candidate_keys_gpu(np.array([[0, 0, 0, 0]], dtype=np.uint16), pairs, args.threads_per_block)

    with open(args.chunk_report_file, "w", encoding="utf-8") as f:
        f.write("SIMON32/64 4-Round Normal MITM PT-Chunk GPU Sweep Report\n")
        f.write("Date: 2026-03-30\n\n")
        f.write("Sweep setup\n")
        f.write(f"- GPU device: {device_name}\n")
        f.write(f"- PT-side chunk size: 2^{args.pt_side_bits}\n")
        f.write(f"- CT-side chunk size: 2^{ct_side_bits}\n")
        f.write(f"- PT chunk start index: {args.pt_chunk_index}\n")
        f.write(f"- PT chunk count requested: {args.pt_chunk_count}\n")
        f.write(f"- CT chunk index: {args.ct_chunk_index}\n")
        f.write(f"- Continue after hit: {args.continue_after_hit}\n")
        f.write(f"- Generated key PT chunk: {chosen_pt_chunk}\n")
        f.write("\nPer-chunk progress\n")

    chunk_results = []
    found_result = None
    found_chunk = None

    for off in range(args.pt_chunk_count):
        pt_chunk = args.pt_chunk_index + off
        res = attack_chunk_gpu(
            pairs=pairs,
            pt_side_bits=args.pt_side_bits,
            ct_side_bits=ct_side_bits,
            pt_chunk_index=pt_chunk,
            ct_chunk_index=args.ct_chunk_index,
            threads_per_block=args.threads_per_block,
        )
        chunk_results.append(res)

        with open(args.chunk_report_file, "a", encoding="utf-8") as f:
            f.write(
                f"- PT chunk {res.pt_chunk_index}: "
                f"total={res.total_seconds:.6f}s "
                f"forward_gpu={res.forward_seconds:.6f}s "
                f"sort_cpu={res.sort_seconds:.6f}s "
                f"backward_gpu={res.backward_seconds:.6f}s "
                f"match_cpu_verify_gpu={res.match_seconds:.6f}s "
                f"matched_bwd={res.matched_backward_candidates} "
                f"candidate_pairs={res.candidate_pairs_checked} "
                f"hit={res.recovered_key is not None}\n"
            )

        if res.recovered_key is not None and found_result is None:
            found_result = res
            found_chunk = pt_chunk
            if not args.continue_after_hit:
                break

    total_elapsed = sum(r.total_seconds for r in chunk_results)
    avg_elapsed = total_elapsed / len(chunk_results)

    pt_side = 1 << args.pt_side_bits
    ct_side = 1 << ct_side_bits
    forward_mids_bytes = pt_side * 4
    backward_mids_bytes = ct_side * 4
    sort_index_bytes = pt_side * 8
    sorted_mids_bytes = pt_side * 4
    unique_mid_bytes_est = pt_side * 4
    unique_meta_bytes_est = pt_side * 16

    lines = [
        "SIMON32/64 4-Round Normal MITM GPU Report",
        "Date: 2026-03-30",
        "",
        "Setup",
        f"- GPU device: {device_name}",
        f"- Demo key (k0,k1,k2,k3): {[hex(x) for x in demo_key]}",
        f"- Number of PT-CT pairs: {args.pairs}",
        f"- PT-side chunk size: 2^{args.pt_side_bits}",
        f"- CT-side chunk size: 2^{ct_side_bits}",
        f"- PT chunk start index: {args.pt_chunk_index}",
        f"- PT chunks attempted: {len(chunk_results)}",
        f"- PT chunk count requested: {args.pt_chunk_count}",
        f"- CT chunk index: {args.ct_chunk_index}",
        f"- Continue after hit: {args.continue_after_hit}",
        f"- Threads per block: {args.threads_per_block}",
        "",
        "Outcome",
        f"- Generated key PT chunk: {chosen_pt_chunk}",
        f"- Found key chunk: {found_chunk if found_chunk is not None else 'not found'}",
        f"- Found key value: {[hex(x) for x in found_result.recovered_key] if found_result else None}",
        f"- Full key verification success: {found_result is not None and found_result.recovered_key == demo_key}",
        f"- Total elapsed over attempted chunks: {format_seconds(total_elapsed)}",
        f"- Average time per attempted chunk: {format_seconds(avg_elapsed)}",
        "",
        "Chunk memory footprint",
        f"- Forward mids host array: {forward_mids_bytes:,} ({gib(forward_mids_bytes):.3f} GiB)",
        f"- Backward mids host array: {backward_mids_bytes:,} ({gib(backward_mids_bytes):.3f} GiB)",
        f"- Forward sort-index array: {sort_index_bytes:,} ({gib(sort_index_bytes):.3f} GiB)",
        f"- Forward sorted-mids array: {sorted_mids_bytes:,} ({gib(sorted_mids_bytes):.3f} GiB)",
        f"- Unique-mid estimate: {unique_mid_bytes_est:,} ({gib(unique_mid_bytes_est):.3f} GiB)",
        f"- Unique metadata estimate (index+count): {unique_meta_bytes_est:,} ({gib(unique_meta_bytes_est):.3f} GiB)",
        f"- Approx transient host total (major arrays): "
        f"{forward_mids_bytes + backward_mids_bytes + sort_index_bytes + sorted_mids_bytes + unique_mid_bytes_est + unique_meta_bytes_est:,} "
        f"({gib(forward_mids_bytes + backward_mids_bytes + sort_index_bytes + sorted_mids_bytes + unique_mid_bytes_est + unique_meta_bytes_est):.3f} GiB)",
        "",
        "Notes",
        "- GPU accelerates midpoint generation on both PT and CT sides.",
        "- Sorting and candidate assembly are CPU-side; key verification batches run on GPU.",
        "- Progress-safe per-chunk logging is written continuously to chunk report file.",
    ]

    with open(args.report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(args.chunk_report_file, "a", encoding="utf-8") as f:
        f.write("\nSummary\n")
        f.write(f"- PT chunks attempted: {len(chunk_results)}\n")
        f.write(f"- Found key chunk: {found_chunk if found_chunk is not None else 'not found'}\n")
        f.write(f"- Found key value: {[hex(x) for x in found_result.recovered_key] if found_result else None}\n")
        f.write(f"- Average chunk time: {avg_elapsed:.6f}s\n")

    print("\n".join(lines))
    print(f"\nReport written to: {args.report_file}")
    print(f"Chunk report written to: {args.chunk_report_file}")


if __name__ == "__main__":
    main()
