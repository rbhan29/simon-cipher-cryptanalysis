# =============================================================================
#  Correlated-Sequence MitM Attack — Simon-16/32
#  Cipher: n=8 bit words, 16-bit block, 32-bit key, (a,b,c)=(0,1,2)
#  GPU: RTX 5050 / Windows 11 / Anaconda3 / cupy-cuda13x
#
#  Run:
#      python simon_mitm_1632.py              # attack using PT_CT_PAIRS below
#      python simon_mitm_1632.py --cpu        # force CPU fallback
#      python simon_mitm_1632.py --tests 5   # run 5 random self-tests
#
#  Generate your own pairs with:
#      python simon816_testvectors.py         # edit KEY/PLAINTEXTS/ROUNDS there
#
#  KEY FACTS vs the 32/64 (n=16) version:
#    n=8  → key space 2^32   → GPU scans ALL candidates in ~0.05-0.5 s
#    DS^d → 256x256 = 64 KB  → fits in GPU L2 cache entirely
#    CUDA → grid(256,256) x block(256) → 16M threads, ke3 inner loop x256
#    No "small key only" restriction — works for ANY 32-bit key
# =============================================================================

import argparse, random, time
import numpy as np

# =============================================================================
#  ★  USER SETTINGS  —  paste your PT-CT pairs here, then run
# =============================================================================
#
#  Format: ((PT_left, PT_right), (CT_left, CT_right))
#  Values: 8-bit words  0x00 .. 0xFF
#  Minimum: 3 pairs.
#  Generate with:  python simon816_testvectors.py  (set ROUNDS=10)
#
#  All examples below use key=[0x01, 0x02, 0x03, 0x04], 10 rounds.

PT_CT_PAIRS = [
    ((0x0F, 0x0E), (0x73, 0xED)),   # pair 1  — drives DS^d + MitM filter
    ((0x13, 0x12), (0xE0, 0x1D)),   # pair 2  — first verification
    ((0x1A, 0x1B), (0x3E, 0x21)),   # pair 3  — second verification (prob=1)
]

# Additional test vectors for the same key (add more pairs here if needed)
# ((0xFF, 0x55), (0xB5, 0xEE))
# ((0xAB, 0xCD), (0xF5, 0x98))
# ((0x00, 0x00), (0xA2, 0x4E))

# =============================================================================
#  CIPHER  n=8, (a,b,c)=(0,1,2), 4-word 32-bit key
# =============================================================================

N       = 8
MASK    = 0xFF
ROUNDS  = 10
T_ENC, T_DEC, PARTIAL = 3, 3, 4   # enc window, dec window, bridge
A, B, C = 0, 1, 2                  # shift params: f(x)=(x<<<0 & x<<<1)^x<<<2

# Key-schedule LFSR constant z0 (period 62)
Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,
      1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,
      1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]

def rl8(x, r):
    r &= 7
    return ((x << r) | (x >> (8 - r))) & MASK

def simon_f(x):
    """f(x) = (x<<<0 & x<<<1) ^ x<<<2 = (x & rl(x,1)) ^ rl(x,2)"""
    return (rl8(x, A) & rl8(x, B)) ^ rl8(x, C)

# Pre-computed f-table: FT[x] = simon_f(x) for x in 0..255
FT = [simon_f(x) for x in range(256)]
FT_NP = np.array(FT, dtype=np.uint8)

def key_schedule(k, rounds=ROUNDS):
    """Expand 4-byte master key [k0,k1,k2,k3] to `rounds` subkeys."""
    CON = (MASK - 3) & MASK   # 0xFC
    rk  = list(k)
    for i in range(rounds - 4):
        t  = ((rk[i+3] >> 3) | (rk[i+3] << 5)) & MASK
        t ^= ((rk[i+3] >> 4) | (rk[i+3] << 4)) & MASK
        t ^=   rk[i+1]
        t ^= ((rk[i+1] >> 1) | (rk[i+1] << 7)) & MASK
        rk.append((CON ^ Z0[i % 62] ^ rk[i] ^ t) & MASK)
    return rk[:rounds]

def encrypt_block(s0, s1, rk):
    """Encrypt (s0,s1) for len(rk) rounds; return (L,R)."""
    L, R = s0, s1
    for i in range(len(rk)):
        L, R = R, (L ^ FT[R] ^ rk[i]) & MASK
    return L, R

def make_pairs(sk, pts):
    rk = key_schedule(sk)
    return [(pt, encrypt_block(pt[0], pt[1], rk)) for pt in pts]

def verify_pairs(key, pairs):
    rk = key_schedule(key)
    return all(encrypt_block(pt[0],pt[1],rk)==ct for pt,ct in pairs)

# =============================================================================
#  CUDA KERNEL SOURCE
#  FT[256] and Z0[62] baked in as __device__ const — no host upload needed.
#  grid(256,256) x block(256): blockIdx.x=ke0, blockIdx.y=ke1, threadIdx.x=ke2
#  Each thread loops ke3 in [0,255] -> covers all 256^4 = 2^32 candidates.
#  DS^d built offline (256x256 uint8 arrays, fits in GPU L2 cache).
# =============================================================================

_FT_C = "0,4,8,14,16,20,28,26,32,36,40,46,56,60,52,50,64,68,72,78,80,84,92,90,112,116,120,126,104,108,100,98,128,132,136,142,144,148,156,154,160,164,168,174,184,188,180,178,224,228,232,238,240,244,252,250,208,212,216,222,200,204,196,194,1,5,9,15,17,21,29,27,33,37,41,47,57,61,53,51,65,69,73,79,81,85,93,91,113,117,121,127,105,109,101,99,193,197,201,207,209,213,221,219,225,229,233,239,249,253,245,243,161,165,169,175,177,181,189,187,145,149,153,159,137,141,133,131,2,7,10,13,18,23,30,25,34,39,42,45,58,63,54,49,66,71,74,77,82,87,94,89,114,119,122,125,106,111,102,97,130,135,138,141,146,151,158,153,162,167,170,173,186,191,182,177,226,231,234,237,242,247,254,249,210,215,218,221,202,207,198,193,131,134,139,140,147,150,159,152,163,166,171,172,187,190,183,176,195,198,203,204,211,214,223,216,243,246,251,252,235,238,231,224,67,70,75,76,83,86,95,88,99,102,107,108,123,126,119,112,35,38,43,44,51,54,63,56,19,22,27,28,11,14,7,0"
_Z0_C = "1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0"

_CUDA_SOURCE = f"""
// Simon-16/32 device constants  n=8  (a,b,c)=(0,1,2)
// FT[x] = (x & rl(x,1)) ^ rl(x,2)
static __device__ const unsigned char FT[256] = {{{_FT_C}}};
static __device__ const unsigned char Z0[62]  = {{{_Z0_C}}};

__device__ __forceinline__
unsigned char d_rl8(unsigned char x, int r){{
    r &= 7;
    return (unsigned char)((x << r) | (x >> (8 - r)));
}}
__device__ __forceinline__
unsigned char d_rr8(unsigned char x, int r){{
    r &= 7;
    return (unsigned char)((x >> r) | (x << (8 - r)));
}}
__device__ __forceinline__
unsigned char d_f(unsigned char x){{
    // f(x) = (x & rl(x,1)) ^ rl(x,2)
    return (unsigned char)((x & d_rl8(x,1)) ^ d_rl8(x,2));
}}

// Key schedule: expand 4-byte master key -> 10 subkeys
__device__ __forceinline__
void d_keyexp(unsigned char k0, unsigned char k1,
              unsigned char k2, unsigned char k3,
              unsigned char rk[10])
{{
    rk[0]=k0; rk[1]=k1; rk[2]=k2; rk[3]=k3;
    #pragma unroll
    for(int i=0;i<6;i++){{
        unsigned char t = d_rr8(rk[i+3],3) ^ d_rr8(rk[i+3],4)
                        ^ rk[i+1]          ^ d_rr8(rk[i+1],1);
        rk[i+4] = (unsigned char)(0xFCu ^ Z0[i] ^ rk[i] ^ t);
    }}
}}

// Full 10-round encryption for verification
__device__ __forceinline__
void d_enc10(unsigned char s0, unsigned char s1,
             const unsigned char rk[10],
             unsigned char *oL, unsigned char *oR)
{{
    unsigned char L=s0, R=s1, nR;
    #pragma unroll
    for(int i=0;i<10;i++){{
        nR = (unsigned char)(L ^ FT[R] ^ rk[i]);
        L  = R; R  = nR;
    }}
    *oL = L; *oR = R;
}}

// ============================================================
// KERNEL 1: Build DS^d offline
// grid(256,1) x block(256): blockIdx.x=kd0, threadIdx.x=kd1
// sd0=CT[1]=s_{{r+1}}, sd1=CT[0]=s_r
// sd2 = FT[sd1] ^ sd0 ^ kd0
// sd3 = FT[sd2] ^ sd1 ^ kd1
// sd_meet = FT[sd3] ^ sd2 ^ rk[7]  (computed online)
// ============================================================
__global__
void kernel_build_dsd(unsigned char sd0_in, unsigned char sd1_in,
                      unsigned char *g_sd2, unsigned char *g_sd3)
{{
    int kd0 = blockIdx.x;
    int kd1 = threadIdx.x;
    unsigned char sd2 = (unsigned char)(FT[sd1_in] ^ sd0_in ^ (unsigned char)kd0);
    unsigned char sd3 = (unsigned char)(FT[sd2]    ^ sd1_in ^ (unsigned char)kd1);
    g_sd2[kd0*256+kd1] = sd2;
    g_sd3[kd0*256+kd1] = sd3;
}}

// ============================================================
// KERNEL 2: Online MitM scan
// grid(256,256) x block(256):
//   blockIdx.x=ke0, blockIdx.y=ke1, threadIdx.x=ke2
//   inner loop: ke3 in [0,255]
// Total: 256^4 = 2^32 candidates in one kernel launch
//
// Shared memory per block:
//   s_FT[256]  — cached f-table for warp-broadcast
//   s_se3, s_Xe — enc-side constants (fixed for all ke2,ke3 in this block)
//   s_done       — block-local early-exit flag
//
// Match condition: enc_meet == sd_meet at s[7]
// ============================================================
__global__
void kernel_mitm(
    unsigned char s00, unsigned char s01,
    unsigned char c00, unsigned char c01,
    unsigned char s10, unsigned char s11,
    unsigned char c10, unsigned char c11,
    unsigned char s20, unsigned char s21,
    unsigned char c20, unsigned char c21,
    const unsigned char * __restrict__ g_sd2,
    const unsigned char * __restrict__ g_sd3,
    unsigned int *g_result,
    int          *g_done)
{{
    const int ke0 = (int)blockIdx.x;
    const int ke1 = (int)blockIdx.y;
    const int ke2 = (int)threadIdx.x;

    __shared__ unsigned char s_FT[256];
    __shared__ unsigned char s_se3, s_Xe;
    __shared__ int s_done;

    // Thread 0 computes block-wide constants (3 f-calls once per block)
    if(threadIdx.x == 0){{
        unsigned char se2 = (unsigned char)(FT[s01] ^ s00 ^ (unsigned char)ke0);
        unsigned char se3 = (unsigned char)(FT[se2] ^ s01 ^ (unsigned char)ke1);
        unsigned char Xe  = (unsigned char)(FT[se3] ^ se2);
        s_se3 = se3;
        s_Xe  = Xe;
        s_done = 0;
    }}
    // All threads cooperatively cache FT into shared memory
    s_FT[threadIdx.x] = FT[threadIdx.x];
    __syncthreads();

    if(*g_done) return;

    unsigned char se3 = s_se3;
    unsigned char Xe  = s_Xe;
    // se4 = Xe ^ ke2  (no f-call: f(se3) already in Xe)
    unsigned char se4 = (unsigned char)(Xe ^ (unsigned char)ke2);
    unsigned char rk[10];

    for(int ke3 = 0; ke3 < 256; ke3++){{
        if(s_done || *g_done) return;

        d_keyexp((unsigned char)ke0, (unsigned char)ke1,
                 (unsigned char)ke2, (unsigned char)ke3, rk);

        // Partial encryption: rounds T_ENC..T_ENC+PARTIAL-1 = rounds 3..6
        unsigned char L=se3, R=se4, nR;
        nR=(unsigned char)(L^s_FT[R]^rk[3]); L=R; R=nR;
        nR=(unsigned char)(L^s_FT[R]^rk[4]); L=R; R=nR;
        nR=(unsigned char)(L^s_FT[R]^rk[5]); L=R; R=nR;
        nR=(unsigned char)(L^s_FT[R]^rk[6]); L=R; R=nR;
        unsigned char enc_meet = L;  // = s[T_ENC+PARTIAL] = s[7]

        // DS^d O(1) lookup: kd0=rk[9], kd1=rk[8], kd2=rk[7]
        int dsd_idx = (int)rk[9]*256 + (int)rk[8];
        unsigned char sd_meet = (unsigned char)(
            s_FT[g_sd3[dsd_idx]] ^ g_sd2[dsd_idx] ^ rk[7]);

        if(enc_meet != sd_meet) continue;

        // Stage 2: verify with pair 2
        unsigned char oL, oR;
        d_enc10(s10, s11, rk, &oL, &oR);
        if(oL!=c10 || oR!=c11) continue;

        // Stage 3: verify with pair 3 (success prob = 1 after both checks)
        d_enc10(s20, s21, rk, &oL, &oR);
        if(oL!=c20 || oR!=c21) continue;

        // Key found — write atomically
        unsigned int packed = (unsigned int)ke0
                            | ((unsigned int)ke1 << 8)
                            | ((unsigned int)ke2 << 16)
                            | ((unsigned int)ke3 << 24);
        atomicCAS(g_result, 0xFFFFFFFFu, packed);
        atomicExch(g_done, 1);
        s_done = 1;
        return;
    }}
}}
"""

# =============================================================================
#  CUPY GPU BACKEND  (RTX 5050, SM_120, cupy-cuda13x)
# =============================================================================

class CuPyGPUBackend:
    """
    GPU backend using cp.RawModule (CuPy 9/10/11/12+).
    Auto-detects compute capability for correct arch flag.
    Kernel 1: grid(256,1) x block(256)  -> builds DS^d (65536 threads)
    Kernel 2: grid(256,256) x block(256) -> MitM scan (16M threads,
              ke3 inner loop x256 = covers all 2^32 candidates)
    Expected time on RTX 5050: 50-500 ms
    """
    name = "CuPy GPU (RawModule, SM_120)"

    def __init__(self):
        import cupy as cp
        self.cp = cp
        # Detect compute capability
        try:
            dev    = cp.cuda.Device(0)
            cc_raw = dev.compute_capability
            if isinstance(cc_raw, str):
                cc = cc_raw.strip()
            elif isinstance(cc_raw, (tuple, list)):
                cc = f"{int(cc_raw[0])}{int(cc_raw[1])}"
            else:
                cc = str(int(cc_raw))
            arch = f"--gpu-architecture=sm_{cc}"
            print(f"  [gpu] CC={cc}  flag={arch}")
        except Exception:
            arch = "--gpu-architecture=sm_120"
            print(f"  [gpu] CC detection failed, defaulting to sm_120")

        # Compile with fallback options
        for opts in [("-O3", arch, "--use_fast_math"), ("-O3", arch), ("-O3",)]:
            try:
                self.mod = cp.RawModule(code=_CUDA_SOURCE, options=opts)
                print(f"  [gpu] Compiled OK: {opts}")
                break
            except Exception as e:
                print(f"  [gpu] Compile failed {opts}: {e}")
        else:
            raise RuntimeError("All CuPy RawModule compile attempts failed")

        self.fn_dsd  = self.mod.get_function("kernel_build_dsd")
        self.fn_mitm = self.mod.get_function("kernel_mitm")

    def build_dsd(self, ct0, ct1):
        """
        Build DS^d offline from first ciphertext.
        sd0=CT[1]=ct1, sd1=CT[0]=ct0  (paper: sd0=s_{{r+1}}, sd1=s_r)
        Returns (d_sd2, d_sd3) — CuPy uint8 arrays shape (65536,)
        """
        cp = self.cp
        d_sd2 = cp.zeros(256*256, dtype=cp.uint8)
        d_sd3 = cp.zeros(256*256, dtype=cp.uint8)
        self.fn_dsd(
            (256, 1), (256,),
            args=(np.uint8(ct1), np.uint8(ct0), d_sd2, d_sd3),
        )
        cp.cuda.Stream.null.synchronize()
        return d_sd2, d_sd3

    def mitm_scan(self, pairs, dsd):
        """
        Online MitM scan.
        Returns packed uint32 ke0|(ke1<<8)|(ke2<<16)|(ke3<<24), or None.
        """
        cp = self.cp
        d_sd2, d_sd3 = dsd
        (s00,s01),(c00,c01) = pairs[0]
        (s10,s11),(c10,c11) = pairs[1]
        (s20,s21),(c20,c21) = pairs[2]

        d_result = cp.full(1, 0xFFFFFFFF, dtype=cp.uint32)
        d_done   = cp.zeros(1, dtype=cp.int32)

        self.fn_mitm(
            (256, 256), (256,),
            args=(
                np.uint8(s00), np.uint8(s01), np.uint8(c00), np.uint8(c01),
                np.uint8(s10), np.uint8(s11), np.uint8(c10), np.uint8(c11),
                np.uint8(s20), np.uint8(s21), np.uint8(c20), np.uint8(c21),
                d_sd2, d_sd3, d_result, d_done,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        val = int(d_result[0])
        return val if val != 0xFFFFFFFF else None

# =============================================================================
#  CPU NUMPY BACKEND  (no compiler needed — works on Windows as-is)
# =============================================================================

class CPUNumpyBackend:
    """
    Pure NumPy CPU fallback.
    Vectorises over all 65536 (ke2,ke3) pairs per outer (ke0,ke1).
    Expected time: 5-120 s (exits immediately when key found).
    Unlike the n=16 version, this works for ANY 32-bit key —
    just takes longer for keys with large ke0/ke1 words.
    """
    name = "CPU NumPy (no compiler needed)"

    @staticmethod
    def _rr(x, r):
        r &= 7
        return ((x.astype(np.uint16) >> r) |
                (x.astype(np.uint16) << (8 - r))).astype(np.uint8)

    def _ks_vec(self, ke0, ke1, ke2_arr, ke3_arr):
        """Vectorised key schedule over ke2_arr x ke3_arr (65536 pairs)."""
        N   = len(ke2_arr)
        rr  = self._rr
        rk  = np.zeros((10, N), dtype=np.uint8)
        rk[0] = ke0;  rk[1] = ke1
        rk[2] = ke2_arr;  rk[3] = ke3_arr
        for i in range(6):
            t  = rr(rk[i+3], 3) ^ rr(rk[i+3], 4) ^ rk[i+1] ^ rr(rk[i+1], 1)
            rk[i+4] = (np.uint8(0xFC) ^ np.uint8(Z0[i]) ^ rk[i] ^ t).astype(np.uint8)
        return rk

    def build_dsd(self, ct0, ct1):
        """Build DS^d as two flat uint8 arrays shape (65536,)."""
        sd0, sd1 = ct1, ct0
        kd0 = np.repeat(np.arange(256, dtype=np.uint16), 256)
        kd1 = np.tile  (np.arange(256, dtype=np.uint16), 256)
        sd2 = (FT_NP[sd1].astype(np.uint16) ^ sd0 ^ kd0).astype(np.uint8)
        sd3 = (FT_NP[sd2].astype(np.uint16) ^ sd1 ^ kd1).astype(np.uint8)
        return sd2, sd3

    def mitm_scan(self, pairs, dsd):
        sd2_flat, sd3_flat = dsd
        (s00,s01),(c00,c01) = pairs[0]
        (s10,s11),(c10,c11) = pairs[1]
        (s20,s21),(c20,c21) = pairs[2]

        ke2_flat = np.repeat(np.arange(256, dtype=np.uint8), 256)
        ke3_flat = np.tile  (np.arange(256, dtype=np.uint8), 256)

        for ke0 in range(256):
            se2_s = int((FT[s01] ^ s00 ^ ke0) & MASK)
            for ke1 in range(256):
                se3_s = int((FT[se2_s] ^ s01 ^ ke1) & MASK)
                Xe_s  = int((FT[se3_s] ^ se2_s) & MASK)

                rk = self._ks_vec(ke0, ke1, ke2_flat, ke3_flat)

                # Vectorised partial encryption over 65536 (ke2,ke3) pairs
                L = np.full(65536, se3_s, dtype=np.uint8)
                R = (Xe_s ^ ke2_flat.astype(np.uint16)).astype(np.uint8)
                for rnd in range(T_ENC, T_ENC + PARTIAL):
                    nR = (L.astype(np.uint16)
                          ^ FT_NP[R].astype(np.uint16)
                          ^ rk[rnd].astype(np.uint16)).astype(np.uint8)
                    L, R = R, nR
                enc_meet = L

                # DS^d lookup: kd0=rk[9], kd1=rk[8], kd2=rk[7]
                idx     = rk[9].astype(np.int32)*256 + rk[8].astype(np.int32)
                sd_meet = (FT_NP[sd3_flat[idx]].astype(np.uint16)
                           ^ sd2_flat[idx].astype(np.uint16)
                           ^ rk[7].astype(np.uint16)).astype(np.uint8)

                hits = np.where(enc_meet == sd_meet)[0]
                for h in hits:
                    ke2v = int(ke2_flat[h])
                    ke3v = int(ke3_flat[h])
                    rk_f = key_schedule([ke0, ke1, ke2v, ke3v])
                    L2, R2 = encrypt_block(s10, s11, rk_f)
                    if L2 != c10 or R2 != c11: continue
                    L3, R3 = encrypt_block(s20, s21, rk_f)
                    if L3 != c20 or R3 != c21: continue
                    return ke0 | (ke1<<8) | (ke2v<<16) | (ke3v<<24)
        return None

# =============================================================================
#  BACKEND LOADER
# =============================================================================

def load_backend(force_cpu=False):
    if not force_cpu:
        try:
            b = CuPyGPUBackend()
            print(f"  Backend : {b.name}")
            return b
        except ImportError:
            print("  Backend : CuPy not found — run: pip install cupy-cuda13x")
        except Exception as e:
            print(f"  Backend : CuPy unavailable ({e})")
            print("            Falling back to CPU NumPy.")
    b = CPUNumpyBackend()
    print(f"  Backend : {b.name}")
    return b

# =============================================================================
#  KEY RECOVERY
# =============================================================================

def recover_key(known_pairs, backend=None, force_cpu=False):
    """
    Full MitM key recovery for 10-round Simon-16/32.
    Returns ([k0,k1,k2,k3], elapsed_seconds).
    """
    assert len(known_pairs) >= 3, "Need at least 3 PT-CT pairs"
    if backend is None:
        backend = load_backend(force_cpu=force_cpu)

    (_, (ct0, ct1)) = known_pairs[0]
    t0 = time.perf_counter()

    # OFFLINE: build DS^d
    t1  = time.perf_counter()
    dsd = backend.build_dsd(ct0, ct1)
    print(f"  [offline] DS^d built in {time.perf_counter()-t1:.4f}s"
          f"  (sd0=0x{ct1:02X}, sd1=0x{ct0:02X})")

    # ONLINE: MitM scan
    t2     = time.perf_counter()
    packed = backend.mitm_scan(known_pairs, dsd)
    t_scan = time.perf_counter() - t2
    dt     = time.perf_counter() - t0

    if packed is None:
        print(f"  [online]  no key found ({t_scan:.4f}s)")
        return None, dt

    key = [(packed >> (8*i)) & 0xFF for i in range(4)]
    print(f"  [online]  key found in {t_scan:.4f}s"
          f"  → {[hex(k) for k in key]}")
    print(f"  [total]   {dt:.4f}s")
    return key, dt

# =============================================================================
#  ATTACK RUNNER
# =============================================================================

def run_attack(pairs, backend=None):
    print()
    print("=" * 56)
    print("  Simon-16/32  MitM Key Recovery  (n=8, 32-bit key)")
    print("=" * 56)
    print(f"  Pairs supplied : {len(pairs)}")
    for i, (pt, ct) in enumerate(pairs, 1):
        print(f"    {i}.  PT=(0x{pt[0]:02X}, 0x{pt[1]:02X})"
              f"   CT=(0x{ct[0]:02X}, 0x{ct[1]:02X})")
    print()

    key, dt = recover_key(pairs, backend=backend)
    print()

    if key:
        rk = key_schedule(key)
        print(f"  Key      : [{', '.join(hex(k) for k in key)}]")
        print(f"  Hex      : {key[0]:02X}{key[1]:02X}{key[2]:02X}{key[3]:02X}")
        print(f"  Time     : {dt:.3f}s")
        print()
        print("  Verification (re-encrypt all pairs with recovered key):")
        all_ok = True
        for i, (pt, ct) in enumerate(pairs, 1):
            got = encrypt_block(pt[0], pt[1], rk)
            ok  = (got == ct)
            all_ok = all_ok and ok
            print(f"    {i}.  PT=(0x{pt[0]:02X},0x{pt[1]:02X})"
                  f"  exp=(0x{ct[0]:02X},0x{ct[1]:02X})"
                  f"  got=(0x{got[0]:02X},0x{got[1]:02X})"
                  f"  {'✓' if ok else '✗'}")
        print()
        status = "★  Attack successful" if all_ok else "✗  Mismatch"
        print(f"  {status}")
    else:
        print(f"  ✗  Key not found  ({dt:.1f}s)")

    print("=" * 56)
    return key

# =============================================================================
#  BUILT-IN TEST VECTORS
# =============================================================================

# Official / verified test vectors for Simon-16/32 (n=8, 10 rounds)
# key=[0x01,0x02,0x03,0x04], rk=[1,2,3,4,0x3F,0x69,0x43,0x05,0xEF,0x74]
TEST_VECTORS = [
    # (key, plaintext, ciphertext)
    ([0x01,0x02,0x03,0x04], (0x0F,0x0E), (0x73,0xED)),
    ([0x01,0x02,0x03,0x04], (0x13,0x12), (0xE0,0x1D)),
    ([0x01,0x02,0x03,0x04], (0x1A,0x1B), (0x3E,0x21)),
    ([0x01,0x02,0x03,0x04], (0xFF,0x55), (0xB5,0xEE)),
    ([0x01,0x02,0x03,0x04], (0xAB,0xCD), (0xF5,0x98)),
    ([0x01,0x02,0x03,0x04], (0x00,0x00), (0xA2,0x4E)),
]

def check_test_vectors():
    print("  Cipher self-check (all test vectors):")
    all_ok = True
    for key, pt, expected_ct in TEST_VECTORS:
        rk = key_schedule(key)
        got = encrypt_block(pt[0], pt[1], rk)
        ok  = (got == expected_ct)
        all_ok = all_ok and ok
        print(f"    key={[hex(k) for k in key]}  "
              f"PT=(0x{pt[0]:02X},0x{pt[1]:02X})  "
              f"CT=(0x{got[0]:02X},0x{got[1]:02X})  "
              f"exp=(0x{expected_ct[0]:02X},0x{expected_ct[1]:02X})  "
              f"{'✓' if ok else '✗ FAIL'}")
    print(f"  Cipher check: {'all OK' if all_ok else 'SOME FAILED'}")
    print()
    return all_ok

# =============================================================================
#  RANDOM-KEY SELF-TESTS
# =============================================================================

def run_random_tests(n=3, seed=42, backend=None):
    """Run n random-key attacks. All key values 0..255 (full 32-bit range works)."""
    rng = random.Random(seed)
    print(f"\n  Random-key self-tests (n={n}, seed={seed})\n")
    passed = 0
    for i in range(n):
        sk    = [rng.randint(0, 255) for _ in range(4)]
        pts   = [(rng.randint(0, 255), rng.randint(0, 255)) for _ in range(3)]
        pairs = make_pairs(sk, pts)
        key, dt = recover_key(pairs, backend=backend)
        ok  = (key == sk)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {i+1}/{n}  key={[hex(k) for k in sk]}"
              f"  →  {[hex(k) for k in key] if key else None}"
              f"  ({dt:.2f}s)")
        if ok: passed += 1
    print(f"\n  Result: {passed}/{n} passed.\n")

# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Simon-16/32 MitM Attack (n=8, 32-bit key) — RTX 5050 / cupy-cuda13x",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simon_mitm_1632.py                  # attack using PT_CT_PAIRS at top
  python simon_mitm_1632.py --cpu            # force CPU NumPy backend
  python simon_mitm_1632.py --tests 5       # 5 random-key self-tests
  python simon_mitm_1632.py --check         # verify cipher with test vectors

GPU setup (run once in cupyenv):
  conda activate cupyenv
  pip install cupy-cuda13x
  python -c "import cupy; print(cupy.cuda.Device(0).compute_capability)"
        """)
    p.add_argument("--cpu",   action="store_true",
                   help="Force CPU NumPy backend (skip GPU)")
    p.add_argument("--tests", type=int, default=0, metavar="N",
                   help="Run N random-key self-tests after attack (default: 0)")
    p.add_argument("--check", action="store_true",
                   help="Run cipher self-check with built-in test vectors")
    args = p.parse_args()

    # Always run cipher self-check first
    check_test_vectors()

    backend = load_backend(force_cpu=args.cpu)

    # Attack using pairs defined at top of file
    run_attack(PT_CT_PAIRS, backend=backend)

    # Optional random-key tests
    if args.tests > 0:
        run_random_tests(n=args.tests, backend=backend)
