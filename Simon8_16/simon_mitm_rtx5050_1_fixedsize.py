# =============================================================================
#  Simon-32/64  Correlated-Sequence MitM Attack
#  (cipher: n=16 bit words, 32-bit block, 64-bit key, (a,b,c)=(8,1,2))
#  RTX 5050 / Windows 11 / Anaconda3 / cupy-cuda13x
#
#  Run:
#      python simon_mitm_rtx5050.py          # uses PT_CT_PAIRS below
#      python simon_mitm_rtx5050.py --cpu    # force CPU NumPy backend
#      python simon_mitm_rtx5050.py --tests 3  # random-key self-tests
#
#  Pairs must come from simon1632_testvectors.py (n=16, 16-bit words).
# =============================================================================

import argparse, os, random, shutil, sys, time
import numpy as np

# =============================================================================
#  ★  USER SETTINGS  —  paste your PT-CT pairs here, then run
# =============================================================================
#
#  Format: ((PT_left, PT_right), (CT_left, CT_right))
#  Values: 16-bit words  0x0000 .. 0xFFFF
#  Minimum: 3 pairs.
#  Generate pairs with:  python simon1632_testvectors.py
#
#  Example below uses key=[0x1001,0x2002,0x3003,0x4004], 10 rounds:

PT_CT_PAIRS = [
    ((0x0f0e, 0x0000), (0x74b8, 0xb2b6)),
    ((0xabcd, 0x1234), (0xe970, 0xc485)),
    ((0xffff, 0x5555), (0xaa7d, 0x270a)),
]

# =============================================================================
#  CIPHER  n=16, (a,b,c)=(8,1,2), 4-word 64-bit key
# =============================================================================

N     = 16
MASK  = 0xFFFF
ROUNDS  = 10
T_ENC, T_DEC, PARTIAL = 3, 3, 4
A, B, C = 8, 1, 2

Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,
      1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,
      1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]

def rl(x, r):
    r &= (N - 1)
    return ((x << r) | (x >> (N - r))) & MASK

def simon_f(x):
    return (rl(x, A) & rl(x, B)) ^ rl(x, C)

def key_schedule(k, rounds=ROUNDS):
    CON = (MASK - 3) & MASK          # 0xFFFC
    rk  = list(k)
    for i in range(rounds - 4):
        t  = ((rk[i+3] >> 3) | (rk[i+3] << (N-3))) & MASK
        t ^= ((rk[i+3] >> 4) | (rk[i+3] << (N-4))) & MASK
        t ^=   rk[i+1]
        t ^= ((rk[i+1] >> 1) | (rk[i+1] << (N-1))) & MASK
        rk.append((CON ^ Z0[i % 62] ^ rk[i] ^ t) & MASK)
    return rk[:rounds]

def encrypt_block(s0, s1, rk):
    L, R = s0, s1
    for i in range(len(rk)):
        L, R = R, (L ^ simon_f(R) ^ rk[i]) & MASK
    return L, R

def make_pairs(sk, pts):
    rk = key_schedule(sk)
    return [(pt, encrypt_block(pt[0], pt[1], rk)) for pt in pts]


def resolve_key_constraints(fixed_bits=0, fixed_value=0, fixed_mask=None, fixed_bit_offset=0):
    """Return normalized (mask, value) constraints over the 64-bit master key."""
    key_mask = (1 << 64) - 1
    bits = int(fixed_bits)
    offset = int(fixed_bit_offset)
    if bits < 0 or bits > 64:
        raise ValueError("--fixed-bits must be in [0, 64]")
    if offset < 0 or offset > 63:
        raise ValueError("--fixed-bit-offset must be in [0, 63]")

    if fixed_mask is None:
        if bits == 0:
            mask = 0
        else:
            if offset + bits > 64:
                raise ValueError("fixed-bit-offset + fixed-bits must be <= 64")
            if bits == 64:
                mask = key_mask
            else:
                mask = ((1 << bits) - 1) << offset
    else:
        mask = int(fixed_mask) & key_mask

    value = int(fixed_value) & mask
    return mask, value


def split_word_constraints(fixed_mask, fixed_value):
    """Return [(k0_mask,k0_value),...,(k3_mask,k3_value)] as 16-bit words."""
    out = []
    for i in range(4):
        shift = 16 * i
        m = (int(fixed_mask) >> shift) & 0xFFFF
        v = (int(fixed_value) >> shift) & 0xFFFF
        out.append((m, v))
    return out

# =============================================================================
#  CUDA KERNEL SOURCE  (16-bit word Simon, FT is a 65536-entry device array)
#  We embed f as an inline device function rather than a precomputed table
#  (65536 entries = 64 KB per thread block shared memory — too large).
# =============================================================================

_CUDA_SOURCE = r"""
static __device__ const unsigned char Z0[62] =
    {1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,
     1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0};

__device__ __forceinline__
unsigned short d_rl16(unsigned short x, int r){
    r &= 15;
    return (unsigned short)((x << r)|(x >> (16-r)));
}
__device__ __forceinline__
unsigned short d_rr16(unsigned short x, int r){
    r &= 15;
    return (unsigned short)((x >> r)|(x << (16-r)));
}
__device__ __forceinline__
unsigned short d_f(unsigned short x){
    return (unsigned short)((d_rl16(x,8) & d_rl16(x,1)) ^ d_rl16(x,2));
}

__device__ __forceinline__
void d_keyexp(unsigned short k0, unsigned short k1,
              unsigned short k2, unsigned short k3,
              unsigned short rk[10])
{
    rk[0]=k0; rk[1]=k1; rk[2]=k2; rk[3]=k3;
    unsigned short CON = (unsigned short)0xFFFC;
    #pragma unroll
    for(int i=0;i<6;i++){
        unsigned short t = d_rr16(rk[i+3],3) ^ d_rr16(rk[i+3],4)
                         ^ rk[i+1]            ^ d_rr16(rk[i+1],1);
        rk[i+4] = (unsigned short)(CON ^ Z0[i] ^ rk[i] ^ t);
    }
}

__device__ __forceinline__
void d_enc10(unsigned short s0, unsigned short s1,
             const unsigned short rk[10],
             unsigned short *oL, unsigned short *oR)
{
    unsigned short L=s0, R=s1, nR;
    #pragma unroll
    for(int i=0;i<10;i++){
        nR=(unsigned short)(L ^ d_f(R) ^ rk[i]);
        L=R; R=nR;
    }
    *oL=L; *oR=R;
}

// ── Kernel 1: build DS^d ────────────────────────────────────────────────────
// Grid (65536,1) x Block (1): one thread per (kd0) row.
// Each thread loops over kd1 in [0,65535] and writes to a flat array.
// This is done in batches to stay within memory.
//
// For RTX 5050 we build DS^d as a device hash:
//   key   = kd0*65536 + kd1  (32-bit index)
//   value = (sd2 << 16) | sd3
// We store in a global array indexed by (kd0*65536+kd1).
// Total: 65536*65536 = 4GB -- NOT FEASIBLE as a full table.
//
// ALTERNATIVE: inline the DS^d lookup during the MitM scan.
// For each candidate key, compute rk[7],rk[8],rk[9] and derive sd_meet
// directly without prebuilding a table.  The DS^d is:
//   sd2 = f(sd1)^sd0^rk[9]
//   sd3 = f(sd2)^sd1^rk[8]
//   sd_meet = f(sd3)^sd2^rk[7]
// This is 3 f-calls per candidate but avoids the 4GB table entirely.

__global__
void kernel_mitm(
    unsigned short s00, unsigned short s01,
    unsigned short c00, unsigned short c01,
    unsigned short s10, unsigned short s11,
    unsigned short c10, unsigned short c11,
    unsigned short s20, unsigned short s21,
    unsigned short c20, unsigned short c21,
    unsigned short sd0_in, unsigned short sd1_in,
    unsigned short k0_mask, unsigned short k0_value,
    unsigned short k1_mask, unsigned short k1_value,
    unsigned short k2_mask, unsigned short k2_value,
    unsigned short k3_mask, unsigned short k3_value,
    unsigned int  *g_result,
    int           *g_done)
{
    // Grid: blockIdx.x = ke0 (0..65535),  blockIdx.y = ke1 batch
    // Each block handles one (ke0, ke1) pair.
    // threadIdx.x = ke2 chunk index (0..255), inner loop ke2 full range.

    const unsigned int ke0 = blockIdx.x;
    const unsigned int ke1 = blockIdx.y * blockDim.y + threadIdx.y;

    if(ke1 >= 65536) return;
    if(*g_done) return;
    if((((unsigned short)ke0) & k0_mask) != k0_value) return;
    if((((unsigned short)ke1) & k1_mask) != k1_value) return;

    __shared__ int s_done;
    if(threadIdx.x == 0 && threadIdx.y == 0) s_done = 0;
    __syncthreads();

    // Step A: 3 f-calls, fixed for this (ke0,ke1)
    unsigned short se2 = (unsigned short)(d_f(s01) ^ s00 ^ (unsigned short)ke0);
    unsigned short se3 = (unsigned short)(d_f(se2) ^ s01 ^ (unsigned short)ke1);
    unsigned short Xe  = (unsigned short)(d_f(se3) ^ se2);

    // Each thread handles a slice of ke2
    unsigned int ke2_start = threadIdx.x * (65536 / blockDim.x);
    unsigned int ke2_end   = ke2_start + (65536 / blockDim.x);

    unsigned short rk[10];

    for(unsigned int ke2 = ke2_start; ke2 < ke2_end; ke2++){
        if(s_done || *g_done) return;
        if((((unsigned short)ke2) & k2_mask) != k2_value) continue;
        unsigned short se4 = (unsigned short)(Xe ^ (unsigned short)ke2);

        for(unsigned int ke3 = 0; ke3 < 65536; ke3++){
            if(s_done || *g_done) return;
            if((((unsigned short)ke3) & k3_mask) != k3_value) continue;

            d_keyexp((unsigned short)ke0,(unsigned short)ke1,
                     (unsigned short)ke2,(unsigned short)ke3, rk);

            // Partial enc: rounds T_ENC..T_ENC+PARTIAL-1
            unsigned short L=se3, R=se4, nR;
            nR=(unsigned short)(L^d_f(R)^rk[3]); L=R; R=nR;
            nR=(unsigned short)(L^d_f(R)^rk[4]); L=R; R=nR;
            nR=(unsigned short)(L^d_f(R)^rk[5]); L=R; R=nR;
            nR=(unsigned short)(L^d_f(R)^rk[6]); L=R; R=nR;
            unsigned short enc_meet = L;

            // Dec-side meet (inline DS^d, 3 f-calls)
            unsigned short sd2=(unsigned short)(d_f(sd1_in)^sd0_in^rk[9]);
            unsigned short sd3=(unsigned short)(d_f(sd2)^sd1_in^rk[8]);
            unsigned short sd_meet=(unsigned short)(d_f(sd3)^sd2^rk[7]);

            if(enc_meet != sd_meet) continue;

            // Verify pair 2
            unsigned short oL,oR;
            d_enc10(s10,s11,rk,&oL,&oR);
            if(oL!=c10||oR!=c11) continue;

            // Verify pair 3
            d_enc10(s20,s21,rk,&oL,&oR);
            if(oL!=c20||oR!=c21) continue;

            // Found
            // Pack as two uint32: lower = ke0|(ke1<<16), upper = ke2|(ke3<<16)
            unsigned int lo = (unsigned int)ke0 | ((unsigned int)ke1 << 16);
            atomicCAS(g_result,   0xFFFFFFFFu, lo);
            unsigned int hi = (unsigned int)ke2 | ((unsigned int)ke3 << 16);
            atomicCAS(g_result+1, 0xFFFFFFFFu, hi);
            atomicExch(g_done, 1);
            s_done = 1;
            return;
        }
    }
}
"""

# =============================================================================
#  NUMBA CUDA KERNEL (fallback GPU path for environments without CuPy NVRTC)
# =============================================================================

try:
    from numba import cuda, uint16, int32
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    cuda = None
    uint16 = None
    int32 = None


if _NUMBA_AVAILABLE:
    @cuda.jit(device=True, inline=True)
    def _rl16_cuda(x, r):
        r &= 15
        return ((x << r) | (x >> (16 - r))) & 0xFFFF

    @cuda.jit(device=True, inline=True)
    def _rr16_cuda(x, r):
        r &= 15
        return ((x >> r) | (x << (16 - r))) & 0xFFFF

    @cuda.jit(device=True, inline=True)
    def _f_cuda(x):
        return (_rl16_cuda(x, 8) & _rl16_cuda(x, 1)) ^ _rl16_cuda(x, 2)

    @cuda.jit(device=True, inline=True)
    def _keyexp10_cuda(k0, k1, k2, k3, rk):
        rk[0] = k0
        rk[1] = k1
        rk[2] = k2
        rk[3] = k3
        con = 0xFFFC
        for i in range(6):
            zbit = 1
            if i == 5:
                zbit = 0
            t = _rr16_cuda(rk[i + 3], 3) ^ _rr16_cuda(rk[i + 3], 4)
            t ^= rk[i + 1] ^ _rr16_cuda(rk[i + 1], 1)
            rk[i + 4] = (con ^ zbit ^ rk[i] ^ t) & 0xFFFF

    @cuda.jit(device=True, inline=True)
    def _enc10_cuda(s0, s1, rk):
        l = s0
        r = s1
        for i in range(10):
            nr = (l ^ _f_cuda(r) ^ rk[i]) & 0xFFFF
            l = r
            r = nr
        return l, r

    @cuda.jit
    def _kernel_mitm_numba(
        s00, s01, c00, c01,
        s10, s11, c10, c11,
        s20, s21, c20, c21,
        sd0_in, sd1_in,
        k0_mask, k0_value,
        k1_mask, k1_value,
        k2_mask, k2_value,
        k3_mask, k3_value,
        g_key, g_done,
    ):
        ke0 = cuda.blockIdx.x
        if g_done[0] != 0:
            return
        if (ke0 & k0_mask) != k0_value:
            return

        tx = cuda.threadIdx.x
        bdx = cuda.blockDim.x
        chunk = 65536 // bdx
        ke2_start = tx * chunk
        ke2_end = ke2_start + chunk

        rk = cuda.local.array(10, dtype=uint16)

        ke1 = cuda.blockIdx.y
        while ke1 < 65536:
            if (ke1 & k1_mask) != k1_value:
                ke1 += cuda.gridDim.y
                continue

            se2 = (_f_cuda(s01) ^ s00 ^ ke0) & 0xFFFF
            se3 = (_f_cuda(se2) ^ s01 ^ ke1) & 0xFFFF
            xe = (_f_cuda(se3) ^ se2) & 0xFFFF

            for ke2 in range(ke2_start, ke2_end):
                if g_done[0] != 0:
                    return
                if (ke2 & k2_mask) != k2_value:
                    continue
                se4 = (xe ^ ke2) & 0xFFFF

                for ke3 in range(65536):
                    if g_done[0] != 0:
                        return
                    if (ke3 & k3_mask) != k3_value:
                        continue

                    _keyexp10_cuda(ke0, ke1, ke2, ke3, rk)

                    l = se3
                    r = se4
                    nr = (l ^ _f_cuda(r) ^ rk[3]) & 0xFFFF
                    l = r
                    r = nr
                    nr = (l ^ _f_cuda(r) ^ rk[4]) & 0xFFFF
                    l = r
                    r = nr
                    nr = (l ^ _f_cuda(r) ^ rk[5]) & 0xFFFF
                    l = r
                    r = nr
                    nr = (l ^ _f_cuda(r) ^ rk[6]) & 0xFFFF
                    l = r
                    r = nr
                    enc_meet = l

                    sd2 = (_f_cuda(sd1_in) ^ sd0_in ^ rk[9]) & 0xFFFF
                    sd3 = (_f_cuda(sd2) ^ sd1_in ^ rk[8]) & 0xFFFF
                    sd_meet = (_f_cuda(sd3) ^ sd2 ^ rk[7]) & 0xFFFF

                    if enc_meet != sd_meet:
                        continue

                    o_l, o_r = _enc10_cuda(s10, s11, rk)
                    if o_l != c10 or o_r != c11:
                        continue
                    o_l, o_r = _enc10_cuda(s20, s21, rk)
                    if o_l != c20 or o_r != c21:
                        continue

                    if cuda.atomic.compare_and_swap(g_done, 0, 1) == 0:
                        g_key[0] = ke0
                        g_key[1] = ke1
                        g_key[2] = ke2
                        g_key[3] = ke3
                    return

            ke1 += cuda.gridDim.y

# =============================================================================
#  CUPY GPU BACKEND
# =============================================================================

class CuPyGPUBackend:
    name = "CuPy GPU (RawModule, n=16)"

    @staticmethod
    def _uniq_paths(paths):
        seen = set()
        out = []
        for p in paths:
            if not p:
                continue
            ap = os.path.abspath(p)
            key = ap.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(ap)
        return out

    @classmethod
    def _candidate_cuda_bins(cls):
        bins = []

        # Conda/venv locations first (common on Windows).
        bins.append(os.path.join(sys.prefix, "Library", "bin"))
        bins.append(os.path.join(sys.prefix, "DLLs"))

        # User-provided CUDA_PATH and versioned CUDA_PATH_V* vars.
        for k, v in os.environ.items():
            ku = k.upper()
            if ku == "CUDA_PATH" or ku.startswith("CUDA_PATH_V"):
                bins.append(os.path.join(v, "bin"))

        # If nvcc is already discoverable, use its folder.
        nvcc = shutil.which("nvcc") or shutil.which("nvcc.exe")
        if nvcc:
            bins.append(os.path.dirname(nvcc))

        # Standard CUDA toolkit install path.
        cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(cuda_root):
            try:
                for vdir in sorted(os.listdir(cuda_root), reverse=True):
                    bins.append(os.path.join(cuda_root, vdir, "bin"))
            except Exception:
                pass

        return [p for p in cls._uniq_paths(bins) if os.path.isdir(p)]

    @classmethod
    def _prepare_windows_cuda_runtime(cls):
        if not sys.platform.startswith("win"):
            return []

        bin_dirs = cls._candidate_cuda_bins()
        if not bin_dirs:
            return []

        # Help CuPy detect toolkit location when CUDA_PATH is unset.
        if "CUDA_PATH" not in os.environ:
            for b in bin_dirs:
                if b.lower().endswith("\\bin"):
                    os.environ["CUDA_PATH"] = os.path.dirname(b)
                    break

        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        path_norm = {os.path.abspath(p).lower() for p in path_parts if p}
        prepend = [b for b in bin_dirs if os.path.abspath(b).lower() not in path_norm]
        if prepend:
            os.environ["PATH"] = os.pathsep.join(prepend + path_parts)

        # On Python 3.8+, explicitly register DLL directories.
        dll_handles = []
        add_dll_dir = getattr(os, "add_dll_directory", None)
        if add_dll_dir is not None:
            for b in bin_dirs:
                try:
                    dll_handles.append(add_dll_dir(b))
                except OSError:
                    pass
        return dll_handles

    def __init__(self, fixed_mask=0, fixed_value=0):
        import cupy as cp
        self.cp = cp
        self.fixed_mask = int(fixed_mask) & ((1 << 64) - 1)
        self.fixed_value = int(fixed_value) & self.fixed_mask
        w = split_word_constraints(self.fixed_mask, self.fixed_value)
        self.k0_mask, self.k0_value = np.uint16(w[0][0]), np.uint16(w[0][1])
        self.k1_mask, self.k1_value = np.uint16(w[1][0]), np.uint16(w[1][1])
        self.k2_mask, self.k2_value = np.uint16(w[2][0]), np.uint16(w[2][1])
        self.k3_mask, self.k3_value = np.uint16(w[3][0]), np.uint16(w[3][1])
        self._dll_handles = self._prepare_windows_cuda_runtime()
        try:
            dev    = cp.cuda.Device(0)
            cc_raw = dev.compute_capability
            if isinstance(cc_raw, str):
                cc = cc_raw.strip()
            elif isinstance(cc_raw, (tuple, list)):
                cc = f"{int(cc_raw[0])}{int(cc_raw[1])}"
            else:
                cc = str(int(cc_raw))
            arch_sm = f"sm_{cc}"
            arch_compute = f"compute_{cc}"
        except Exception:
            arch_sm = "sm_120"
            arch_compute = "compute_120"

        attempts = []
        self.compile_backend = "unknown"
        self.compile_opts = tuple()

        opt_sets = {
            "nvrtc": [
                (f"--gpu-architecture={arch_compute}", "--use_fast_math"),
                (f"--gpu-architecture={arch_compute}",),
                tuple(),
            ],
            "nvcc": [
                (f"-arch={arch_sm}", "-O3", "--use_fast_math"),
                (f"-arch={arch_sm}", "-O3"),
                (f"-arch={arch_sm}",),
                tuple(),
            ],
        }

        for backend in ("nvrtc", "nvcc"):
            for opts in opt_sets[backend]:
                try:
                    kwargs = dict(code=_CUDA_SOURCE, options=opts)
                    if backend == "nvcc":
                        kwargs["backend"] = "nvcc"
                    mod = cp.RawModule(**kwargs)
                    fn_mitm = mod.get_function("kernel_mitm")
                    self.mod = mod
                    self.fn_mitm = fn_mitm
                    self.compile_backend = backend
                    self.compile_opts = opts
                    break
                except Exception as e:
                    attempts.append(f"{backend} {opts}: {e}")
            if self.compile_backend != "unknown":
                break
        else:
            tail = attempts[-2:] if attempts else ["no compiler attempts"]
            raise RuntimeError("RawModule compile failed: " + " | ".join(tail))

    def build_dsd(self, ct0, ct1):
        # For n=16, DS^d is computed inline in the kernel
        # Return the CT values for the kernel to use directly
        return (ct0, ct1)

    def mitm_scan(self, pairs, dsd):
        cp   = self.cp
        ct0, ct1 = dsd    # sd1=ct0 (s_r), sd0=ct1 (s_{r+1})
        sd0  = np.uint16(ct1)
        sd1  = np.uint16(ct0)

        (s00,s01),(c00,c01) = pairs[0]
        (s10,s11),(c10,c11) = pairs[1]
        (s20,s21),(c20,c21) = pairs[2]

        d_result = cp.full(2, 0xFFFFFFFF, dtype=cp.uint32)
        d_done   = cp.zeros(1, dtype=cp.int32)

        # Grid: (65536 ke0 values) x (ke1 in batches of 256)
        # Block: (256 threads handle ke2 slices) x (1 ke1 per block row)
        # ke3 is iterated inside each thread
        # NOTE: This kernel is illustrative; full 2^64 scan needs HPC GPU time.
        # For demo, limit ke0 range to what can run in reasonable time.
        # Full scan requires dedicated GPU cluster.
        self.fn_mitm(
            (65536, 256), (256, 1),
            args=(np.uint16(s00), np.uint16(s01), np.uint16(c00), np.uint16(c01),
                  np.uint16(s10), np.uint16(s11), np.uint16(c10), np.uint16(c11),
                  np.uint16(s20), np.uint16(s21), np.uint16(c20), np.uint16(c21),
                  sd0, sd1,
                  self.k0_mask, self.k0_value,
                  self.k1_mask, self.k1_value,
                  self.k2_mask, self.k2_value,
                  self.k3_mask, self.k3_value,
                  d_result, d_done),
        )
        cp.cuda.Stream.null.synchronize()

        if int(d_result[0]) == 0xFFFFFFFF:
            return None
        lo = int(d_result[0]); hi = int(d_result[1])
        ke0 = lo & 0xFFFF; ke1 = (lo >> 16) & 0xFFFF
        ke2 = hi & 0xFFFF; ke3 = (hi >> 16) & 0xFFFF
        return [ke0, ke1, ke2, ke3]


class NumbaGPUBackend:
    name = "Numba CUDA (n=16)"

    def __init__(self, fixed_mask=0, fixed_value=0):
        if not _NUMBA_AVAILABLE:
            raise RuntimeError("Numba CUDA not available")
        if not cuda.is_available():
            raise RuntimeError("Numba CUDA runtime unavailable")
        self.fixed_mask = int(fixed_mask) & ((1 << 64) - 1)
        self.fixed_value = int(fixed_value) & self.fixed_mask
        w = split_word_constraints(self.fixed_mask, self.fixed_value)
        self.k0_mask, self.k0_value = np.uint16(w[0][0]), np.uint16(w[0][1])
        self.k1_mask, self.k1_value = np.uint16(w[1][0]), np.uint16(w[1][1])
        self.k2_mask, self.k2_value = np.uint16(w[2][0]), np.uint16(w[2][1])
        self.k3_mask, self.k3_value = np.uint16(w[3][0]), np.uint16(w[3][1])

    def build_dsd(self, ct0, ct1):
        return (ct0, ct1)

    def mitm_scan(self, pairs, dsd):
        ct0, ct1 = dsd
        sd0 = np.uint16(ct1)
        sd1 = np.uint16(ct0)

        (s00, s01), (c00, c01) = pairs[0]
        (s10, s11), (c10, c11) = pairs[1]
        (s20, s21), (c20, c21) = pairs[2]

        d_key = cuda.to_device(np.array([0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF], dtype=np.uint16))
        d_done = cuda.to_device(np.array([0], dtype=np.int32))

        # Same mapping as the CuPy kernel: blockIdx.x=ke0, blockIdx.y=ke1.
        _kernel_mitm_numba[(65536, 256), (256, 1)](
            np.uint16(s00), np.uint16(s01), np.uint16(c00), np.uint16(c01),
            np.uint16(s10), np.uint16(s11), np.uint16(c10), np.uint16(c11),
            np.uint16(s20), np.uint16(s21), np.uint16(c20), np.uint16(c21),
            sd0, sd1,
            self.k0_mask, self.k0_value,
            self.k1_mask, self.k1_value,
            self.k2_mask, self.k2_value,
            self.k3_mask, self.k3_value,
            d_key, d_done,
        )
        cuda.synchronize()

        done = int(d_done.copy_to_host()[0])
        if done == 0:
            return None
        key = d_key.copy_to_host().astype(np.uint16).tolist()
        return [int(key[0]), int(key[1]), int(key[2]), int(key[3])]

# =============================================================================
#  CPU NUMPY BACKEND  (n=16, 16-bit words)
#  Note: full 2^64 key space is too large for CPU.
#  This backend is practical only for DEMO with small/known key ranges.
# =============================================================================

class CPUNumpyBackend:
    name = "CPU NumPy (n=16)"

    def build_dsd(self, ct0, ct1):
        return (ct0, ct1)     # DS^d computed inline during scan

    def mitm_scan(self, pairs, dsd):
        ct0, ct1 = dsd
        sd0, sd1 = ct1, ct0  # sd0=s_{r+1}=CT[1], sd1=s_r=CT[0]

        (s00,s01),(c00,c01) = pairs[0]
        (s10,s11),(c10,c11) = pairs[1]
        (s20,s21),(c20,c21) = pairs[2]

        # For each (ke0,ke1): vectorise over ALL ke2 in [0,65535],
        # then loop ke3. This keeps memory at 65536 x uint16 = 128 KB per row.
        # Total ops: 65536^2 * 65536 = 2^48 — only feasible for demo if key
        # words are small.  We scan the full (ke0,ke1) space but break early.

        ke2_arr = np.arange(65536, dtype=np.uint32)

        def f_vec(x):
            return ((((x << 8)|(x >> 8)) & 0xFFFF &
                     ((x << 1)|(x >> 15)) & 0xFFFF) ^
                    (((x << 2)|(x >> 14)) & 0xFFFF)).astype(np.uint32)

        for ke0 in range(65536):
            se2 = int((simon_f(s01) ^ s00 ^ ke0) & MASK)
            for ke1 in range(65536):
                se3 = int((simon_f(se2) ^ s01 ^ ke1) & MASK)
                Xe  = int((simon_f(se3) ^ se2) & MASK)

                for ke3 in range(65536):
                    # Vectorised over all ke2
                    rk = [0]*10
                    # key_schedule vectorised
                    rk[0]=ke0; rk[1]=ke1
                    # We need ke2 as array -> batch over ke2 inside ke3 loop
                    # Instead: loop ke2 outside ke3 for clarity
                    pass

                # Fallback: plain loop (too slow for full space, demo only)
                for ke2 in range(65536):
                    se4 = int((Xe ^ ke2) & MASK)
                    for ke3 in range(65536):
                        rk = key_schedule([ke0, ke1, ke2, ke3])
                        L, R = se3, se4
                        for rnd in range(T_ENC, T_ENC + PARTIAL):
                            L, R = R, (L ^ simon_f(R) ^ rk[rnd]) & MASK
                        enc_meet = L

                        sd2 = (simon_f(sd1) ^ sd0 ^ rk[9]) & MASK
                        sd3 = (simon_f(sd2) ^ sd1 ^ rk[8]) & MASK
                        sd_meet = (simon_f(sd3) ^ sd2 ^ rk[7]) & MASK

                        if enc_meet != sd_meet:
                            continue

                        L2,R2 = encrypt_block(s10,s11,rk)
                        if L2!=c10 or R2!=c11: continue
                        L3,R3 = encrypt_block(s20,s21,rk)
                        if L3!=c20 or R3!=c21: continue
                        return [ke0, ke1, ke2, ke3]
        return None

# =============================================================================
#  PRACTICAL CPU BACKEND  (uses numpy vectorisation over ke2, loops ke0,ke1,ke3)
# =============================================================================

class CPUPracticalBackend:
    """
    Practical CPU MitM for n=16.
    Vectorises the ke2 axis (65536 candidates at once) using NumPy.
    Inner loop: ke3 in [0,65535] per (ke0,ke1,ke2_batch).
    Total time on laptop: impractical for random keys (2^64 space),
    but works correctly for demo/verification purposes when key words are small
    or a limited key range is scanned.
    For full key recovery on n=16, GPU is required.
    """
    name = "CPU NumPy vectorised (n=16, limited scan)"

    def __init__(self, fixed_mask=0, fixed_value=0):
        self.fixed_mask = int(fixed_mask) & ((1 << 64) - 1)
        self.fixed_value = int(fixed_value) & self.fixed_mask
        w = split_word_constraints(self.fixed_mask, self.fixed_value)
        self.k0_mask, self.k0_value = w[0]
        self.k1_mask, self.k1_value = w[1]
        self.k2_mask, self.k2_value = w[2]
        self.k3_mask, self.k3_value = w[3]

    @staticmethod
    def _rr16(x, r):
        r &= 15
        return ((x.astype(np.uint32) >> r) |
                (x.astype(np.uint32) << (16 - r))).astype(np.uint16)

    @staticmethod
    def _f_vec(x):
        """Vectorised simon_f for numpy arrays of uint16."""
        x32 = x.astype(np.uint32)
        rl8  = ((x32 << 8) | (x32 >> 8)) & 0xFFFF
        rl1  = ((x32 << 1) | (x32 >> 15)) & 0xFFFF
        rl2  = ((x32 << 2) | (x32 >> 14)) & 0xFFFF
        return ((rl8 & rl1) ^ rl2).astype(np.uint16)

    def _ks_vec(self, ke0, ke1, ke2_arr, ke3):
        """Key schedule vectorised over ke2_arr (uint16 array), scalar ke3."""
        N = len(ke2_arr)
        CON = np.uint16(MASK - 3)
        rk  = np.zeros((10, N), dtype=np.uint32)
        rk[0] = ke0; rk[1] = ke1
        rk[2] = ke2_arr.astype(np.uint32)
        rk[3] = ke3

        def rr(x, r):
            r &= 15
            return ((x >> r) | (x << (16 - r))) & 0xFFFF

        for i in range(6):
            t = (rr(rk[i+3],3) ^ rr(rk[i+3],4) ^
                 rk[i+1] ^ rr(rk[i+1],1)) & 0xFFFF
            rk[i+4] = (int(CON) ^ int(Z0[i]) ^ rk[i] ^ t) & 0xFFFF
        return rk

    def build_dsd(self, ct0, ct1):
        return (ct0, ct1)

    def mitm_scan(self, pairs, dsd):
        ct0, ct1 = dsd
        sd0, sd1 = ct1, ct0

        (s00,s01),(c00,c01) = pairs[0]
        (s10,s11),(c10,c11) = pairs[1]
        (s20,s21),(c20,c21) = pairs[2]

        ke2_arr = np.arange(65536, dtype=np.uint16)

        for ke0 in range(65536):
            if (ke0 & self.k0_mask) != self.k0_value:
                continue
            se2 = int((simon_f(s01) ^ s00 ^ ke0) & MASK)
            for ke1 in range(65536):
                if (ke1 & self.k1_mask) != self.k1_value:
                    continue
                se3 = int((simon_f(se2) ^ s01 ^ ke1) & MASK)
                Xe  = int((simon_f(se3) ^ se2) & MASK)

                # se4[ke2] = Xe ^ ke2 for all ke2 at once
                se4_arr = (Xe ^ ke2_arr.astype(np.uint32)).astype(np.uint16)

                for ke3 in range(65536):
                    if (ke3 & self.k3_mask) != self.k3_value:
                        continue
                    rk = self._ks_vec(ke0, ke1, ke2_arr, ke3)

                    # Partial encryption (vectorised over ke2)
                    L = np.full(65536, se3, dtype=np.uint32)
                    R = se4_arr.astype(np.uint32)

                    for rnd in range(T_ENC, T_ENC + PARTIAL):
                        fR = self._f_vec(R.astype(np.uint16)).astype(np.uint32)
                        nR = (L ^ fR ^ rk[rnd]) & 0xFFFF
                        L, R = R, nR
                    enc_meet = L.astype(np.uint16)

                    # Inline DS^d
                    fsd1 = simon_f(sd1)
                    sd2_arr = (fsd1 ^ sd0 ^ rk[9]) & 0xFFFF
                    fsd2    = self._f_vec(sd2_arr.astype(np.uint16)).astype(np.uint32)
                    sd3_arr = (fsd2 ^ sd1 ^ rk[8]) & 0xFFFF
                    fsd3    = self._f_vec(sd3_arr.astype(np.uint16)).astype(np.uint32)
                    sd_meet = (fsd3 ^ sd2_arr ^ rk[7]).astype(np.uint16)

                    hits = np.where(enc_meet == sd_meet)[0]
                    for h in hits:
                        ke2v = int(ke2_arr[h])
                        if (ke2v & self.k2_mask) != self.k2_value:
                            continue
                        rk_f = key_schedule([ke0, ke1, ke2v, ke3])
                        L2,R2 = encrypt_block(s10,s11,rk_f)
                        if L2!=c10 or R2!=c11: continue
                        L3,R3 = encrypt_block(s20,s21,rk_f)
                        if L3!=c20 or R3!=c21: continue
                        return [ke0, ke1, ke2v, ke3]
        return None

# =============================================================================
#  BACKEND LOADER
# =============================================================================

def load_backend(force_cpu=False, fixed_mask=0, fixed_value=0):
    if not force_cpu:
        try:
            b = NumbaGPUBackend(fixed_mask=fixed_mask, fixed_value=fixed_value)
            print(f"  Backend : {b.name}")
            return b
        except Exception as e:
            print(f"  Backend : Numba CUDA unavailable ({e})")
        try:
            b = CuPyGPUBackend(fixed_mask=fixed_mask, fixed_value=fixed_value)
            print(f"  Backend : {b.name}")
            print(f"  Compile : {b.compile_backend}")
            return b
        except ImportError:
            print("  Backend : CuPy not found — pip install cupy-cuda13x")
        except Exception as e:
            print(f"  Backend : CuPy unavailable ({e})")
    b = CPUPracticalBackend(fixed_mask=fixed_mask, fixed_value=fixed_value)
    print(f"  Backend : {b.name}")
    print("  Note    : n=16 full key space (2^64) requires GPU for full search.")
    print("            CPU backend scans keys in order — finds key if it's small.")
    return b

# =============================================================================
#  KEY RECOVERY
# =============================================================================

def recover_key(known_pairs, backend=None, force_cpu=False):
    assert len(known_pairs) >= 3, "Need at least 3 PT-CT pairs"
    if backend is None:
        backend = load_backend(force_cpu=force_cpu)
    (_, (ct0, ct1)) = known_pairs[0]
    t0  = time.perf_counter()
    dsd = backend.build_dsd(ct0, ct1)
    key = backend.mitm_scan(known_pairs, dsd)
    dt  = time.perf_counter() - t0
    return key, dt

# =============================================================================
#  ATTACK FROM USER PAIRS
# =============================================================================

def run_attack(pairs, backend=None, fixed_mask=0, fixed_value=0):
    print()
    print("=" * 54)
    print("  Simon-32/64  MitM Key Recovery  (n=16, 64-bit key)")
    print("=" * 54)
    print(f"  Pairs supplied : {len(pairs)}")
    if fixed_mask:
        print(f"  Key filter     : mask=0x{fixed_mask:016X} value=0x{fixed_value:016X}"
              f" ({fixed_mask.bit_count()} fixed bits)")
    for i,(pt,ct) in enumerate(pairs,1):
        print(f"    {i}. PT=(0x{pt[0]:04X},0x{pt[1]:04X})"
              f"  CT=(0x{ct[0]:04X},0x{ct[1]:04X})")
    print()

    key, dt = recover_key(pairs, backend=backend)
    print()

    if key:
        rk = key_schedule(key)
        print(f"  Key found  : [{', '.join(hex(k) for k in key)}]")
        print(f"  Time       : {dt:.3f}s")
        print()
        print("  Verification:")
        all_ok = True
        for i,(pt,ct) in enumerate(pairs,1):
            got = encrypt_block(pt[0], pt[1], rk)
            ok  = (got == ct)
            all_ok = all_ok and ok
            print(f"    {i}. PT=(0x{pt[0]:04X},0x{pt[1]:04X})"
                  f"  expected (0x{ct[0]:04X},0x{ct[1]:04X})"
                  f"  got (0x{got[0]:04X},0x{got[1]:04X})"
                  f"  {'✓' if ok else '✗'}")
        print()
        print(f"  {'★  Attack successful' if all_ok else '✗  Mismatch'}")
    else:
        print(f"  ✗  Key not found in {dt:.1f}s")
        print("     For n=16 with large keys, use GPU backend (--gpu)")

    print("=" * 54)
    return key

# =============================================================================
#  RANDOM-KEY SELF-TESTS
# =============================================================================

def run_random_tests(n=3, seed=42, backend=None):
    """
    Self-tests using SMALL keys (words in 0..255) so CPU scan completes fast.
    For large 16-bit keys, GPU is needed.
    """
    rng = random.Random(seed)
    print(f"\n  Random-key self-tests (n={n}, seed={seed})")
    print("  Keys use small words (0x00-0xFF) for CPU tractability.\n")
    passed = 0
    for i in range(n):
        # Use small key words so CPU scan finds them quickly
        sk  = [rng.randint(0, 255) for _ in range(4)]
        pts = [(rng.randint(0, 0xFFFF), rng.randint(0, 0xFFFF))
               for _ in range(3)]
        pairs = make_pairs(sk, pts)
        key, dt = recover_key(pairs, backend=backend)
        ok  = (key == sk)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] key=[{','.join(hex(k) for k in sk)}]"
              f"  recovered={[hex(k) for k in key] if key else None}"
              f"  t={dt:.2f}s")
        if ok: passed += 1
    print(f"\n  {passed}/{n} passed.\n")

# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Simon-32/64 MitM (n=16) — RTX 5050 / cupy-cuda13x")
    p.add_argument("--cpu",   action="store_true",
                   help="Force CPU NumPy backend")
    p.add_argument("--tests", type=int, default=0, metavar="N",
                   help="Run N random-key self-tests (default: 0)")
    p.add_argument("--fixed-bits", type=int, default=0, metavar="N",
                   help="Fix N key bits (default: 0)")
    p.add_argument("--fixed-bit-offset", type=int, default=0, metavar="O",
                   help="Bit offset for --fixed-bits mask (default: 0)")
    p.add_argument("--fixed-value", type=lambda x: int(x, 0), default=0,
                   help="Value for fixed bits/mask (default: 0)")
    p.add_argument("--fixed-mask", type=lambda x: int(x, 0), default=None,
                   help="Explicit 64-bit mask; overrides --fixed-bits")
    args = p.parse_args()

    fixed_mask, fixed_value = resolve_key_constraints(
        fixed_bits=args.fixed_bits,
        fixed_value=args.fixed_value,
        fixed_mask=args.fixed_mask,
        fixed_bit_offset=args.fixed_bit_offset,
    )

    backend = load_backend(
        force_cpu=args.cpu,
        fixed_mask=fixed_mask,
        fixed_value=fixed_value,
    )

    run_attack(PT_CT_PAIRS, backend=backend,
               fixed_mask=fixed_mask, fixed_value=fixed_value)

    if args.tests > 0:
        if fixed_mask:
            print("  Note    : --tests skipped when key filter is active.")
            sys.exit(0)
        run_random_tests(n=args.tests, backend=backend)
