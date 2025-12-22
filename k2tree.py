from __future__ import annotations

"""
k^2-tree builder and query helpers (CPU + CuPy GPU).

Quick usage:
  from k2tree import K2Tree
  k2 = K2Tree.from_adjdict(adj, N=N, use_gpu=True)
  hits = k2.adj_batch(u, v)
  k2.save_npz("graph_k2tree.npz")
  k2_cpu = K2Tree.load_npz("graph_k2tree.npz", use_gpu=False)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - handled at runtime
    cp = None


DEFAULT_K = 2
DEFAULT_S_WORDS = 1024

_KERNEL_CACHE: dict[tuple[int, int], "cp.RawKernel"] = {}

__all__ = [
    "K2Tree",
    "DEFAULT_K",
    "DEFAULT_S_WORDS",
    "build_k2tree_from_adjdict",
    "build_k2tree_from_edges",
    "build_k2tree_from_adjacency_matrix",
    "build_rank_tables_u32",
    "clear_kernel_cache",
    "get_k2tree_kernel",
    "k2tree_adj_cpu",
    "k2tree_trace_cpu",
    "k2tree_upload_from_adjdict",
]


def _require_cupy() -> "cp":
    if cp is None:
        raise RuntimeError("CuPy is required for GPU k2-tree operations.")
    return cp


def clear_kernel_cache() -> None:
    _KERNEL_CACHE.clear()


def _k2tree_params(N: int, k: int) -> tuple[int, int]:
    """Return (H, Npad) where Npad = k**H >= N. Logical padding; cells outside N are 0."""
    assert N >= 0
    if N <= 1:
        return 0, 1
    H = 0
    Npad = 1
    while Npad < N:
        Npad *= k
        H += 1
    return H, Npad


def _pack_bits_to_u32(bit_list_0_1: Iterable[int]) -> np.ndarray:
    """Pack Python 0/1 bits (LSB-first within each 32-bit word) into np.uint32 words."""
    bits = list(bit_list_0_1)
    n_bits = len(bits)
    n_words = (n_bits + 31) // 32
    words = np.zeros(n_words, dtype=np.uint32)
    for i, b in enumerate(bits):
        if b:
            words[i >> 5] |= np.uint32(1) << np.uint32(i & 31)
    return words


def _normalize_edges(
    edges: Iterable[Tuple[int, int]],
    N: int | None,
) -> tuple[list[tuple[int, int]], int]:
    edges_list: list[tuple[int, int]] = []
    max_id = -1
    for u, v in edges:
        u_i = int(u)
        v_i = int(v)
        edges_list.append((u_i, v_i))
        if N is None:
            if u_i > max_id:
                max_id = u_i
            if v_i > max_id:
                max_id = v_i
    if N is None:
        N = max_id + 1 if max_id >= 0 else 0
    return edges_list, N


def _build_k2tree_from_edges_list(
    edges: List[Tuple[int, int]],
    N: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    H, Npad = _k2tree_params(N, k)
    K2 = k * k

    # Degenerate case: N<=1 => no meaningful internal structure
    if H == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32), H, Npad, N, 0

    # masks[(level, br, bc)] = bitmask of which children are 1 at that node.
    masks: dict[tuple[int, int, int], int] = {}

    for u0, v0 in edges:
        u = int(u0)
        v = int(v0)
        if u < 0 or u >= N or v < 0 or v >= N:
            continue
        br = 0
        bc = 0
        size = Npad
        for level in range(H):  # includes level==H-1 which corresponds to L bits
            block = size // k
            dr = u // block
            dc = v // block
            j = dr * k + dc  # row-major child index
            key = (level, br, bc)
            masks[key] = masks.get(key, 0) | (1 << j)

            # descend
            u = u % block
            v = v % block
            br = br * k + dr
            bc = bc * k + dc
            size = block

    # Pack in level-order expansion.
    T_bits: list[int] = []
    L_bits: list[int] = []

    current_nodes = [(0, 0)]  # list of (br, bc) at this level, in stored order
    for level in range(H):
        next_nodes = []

        for (br, bc) in current_nodes:
            mask = masks.get((level, br, bc), 0)

            # append this node's K2 child bits in row-major (j=0..K2-1)
            if level < H - 1:
                for j in range(K2):
                    T_bits.append((mask >> j) & 1)
            else:
                for j in range(K2):
                    L_bits.append((mask >> j) & 1)

            # expand only for internal levels (children become nodes at next level)
            if level < H - 1 and mask != 0:
                # enumerate 1-bits j in ascending order
                m = mask
                while m:
                    lsb = m & -m
                    j = (lsb.bit_length() - 1)
                    dr = j // k
                    dc = j - dr * k
                    next_nodes.append((br * k + dr, bc * k + dc))
                    m ^= lsb

        current_nodes = next_nodes

    # invariants / sanity checks
    assert (len(T_bits) % K2) == 0

    T_words = _pack_bits_to_u32(T_bits)
    L_words = _pack_bits_to_u32(L_bits)
    return T_words, L_words, H, Npad, N, len(T_bits)


def _verify_adjdict_full(
    adj: Dict[int, List[int]],
    T_words_h: np.ndarray,
    L_words_h: np.ndarray,
    super_h: np.ndarray,
    sub_h: np.ndarray,
    N_used: int,
    H: int,
    Npad: int,
    T_nbits: int,
    k: int,
    s_words: int,
    progress_every: int | None = 1000,
) -> None:
    v_all = np.arange(N_used, dtype=np.int32)
    if progress_every:
        print(f"verify: N={N_used} rows")
    for u in range(N_used):
        truth = np.zeros(N_used, dtype=np.uint8)
        vs = adj.get(int(u), [])
        for v in vs:
            v_i = int(v)
            if 0 <= v_i < N_used:
                truth[v_i] = 1

        got = np.fromiter(
            (
                k2tree_adj_cpu(
                    T_words_h,
                    L_words_h,
                    super_h,
                    sub_h,
                    int(u),
                    int(vv),
                    N_used,
                    H,
                    Npad,
                    T_nbits,
                    k=k,
                    s_words=s_words,
                )
                for vv in v_all
            ),
            count=N_used,
            dtype=np.uint8,
        )
        mismatch = np.nonzero(truth != got)[0]
        if mismatch.size:
            v0 = int(mismatch[0])
            raise AssertionError(
                f"k2-tree verify failed at u={u} v={v0} truth={int(truth[v0])} got={int(got[v0])}"
            )
        if progress_every and (u + 1) % progress_every == 0:
            print(f"verify: {u + 1}/{N_used} rows")
    if progress_every:
        print("verify: done")


def build_k2tree_from_edges(
    edges: Iterable[Tuple[int, int]],
    N: int | None = None,
    k: int = DEFAULT_K,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Build k^2-tree bitvectors (T_words, L_words) from an iterable of (u, v) edges.
    Returns: (T_words_u32, L_words_u32, H, Npad, N_used, T_nbits)
    """
    edges_list, N_used = _normalize_edges(edges, N)
    return _build_k2tree_from_edges_list(edges_list, N_used, k)


def build_k2tree_from_adjacency_matrix(
    matrix: Any,
    N: int | None = None,
    k: int = DEFAULT_K,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Build k^2-tree bitvectors (T_words, L_words) from a dense or sparse adjacency matrix.
    For sparse inputs, a .tocoo() method is used when available.
    Returns: (T_words_u32, L_words_u32, H, Npad, N_used, T_nbits)
    """
    if hasattr(matrix, "tocoo"):
        coo = matrix.tocoo()
        rows = np.asarray(coo.row)
        cols = np.asarray(coo.col)
        shape = coo.shape
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError("matrix must be 2D")
        rows, cols = np.nonzero(dense)
        shape = dense.shape

    if shape[0] != shape[1] and N is None:
        raise ValueError("matrix must be square unless N is provided")
    if N is None:
        N = int(shape[0])
    if N < shape[0] or N < shape[1]:
        raise ValueError("N must be >= matrix shape")

    edges = list(zip(rows.tolist(), cols.tolist()))
    return build_k2tree_from_edges(edges, N=N, k=k)

def build_k2tree_from_adjdict(
    adj: Dict[int, List[int]],
    N: int | None = None,
    k: int = DEFAULT_K,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Build k^2-tree bitvectors (T_words, L_words) from a directed adjacency dict.
    Layout contract:
      - logical padding to Npad = k**H >= N
      - level-order, expand only 1-bits
      - row-major child order j=dr*k+dc
      - T stores internal levels (bits from level 0..H-2), L stores leaf bits (from level H-1 nodes)
    Returns: (T_words_u32, L_words_u32, H, Npad, N_used, T_nbits)
    """
    max_id = -1 if N is None else 0
    edges: list[tuple[int, int]] = []
    for u, vs in adj.items():
        u_i = int(u)
        for v in vs:
            v_i = int(v)
            edges.append((u_i, v_i))
            if N is None:
                if u_i >= 0 and u_i > max_id:
                    max_id = u_i
                if v_i >= 0 and v_i > max_id:
                    max_id = v_i
    if N is None:
        N = max_id + 1 if max_id >= 0 else 0
    return _build_k2tree_from_edges_list(edges, N, k)


def build_rank_tables_u32(T_words_host_u32: np.ndarray, s_words: int = DEFAULT_S_WORDS) -> tuple[np.ndarray, np.ndarray]:
    """
    Build rank support for packed bits in uint32 words.
    Returns (super_u32, sub_u16) as host arrays.
    - super[s] = popcount up to the start of superblock s
    - sub[w]   = popcount from start of superblock up to word w (exclusive)
    """
    assert T_words_host_u32.dtype == np.uint32
    n_words = int(T_words_host_u32.size)
    n_super = (n_words + s_words - 1) // s_words

    super_u32 = np.zeros(n_super + 1, dtype=np.uint32)
    sub_u16 = np.zeros(n_words, dtype=np.uint16)

    running = 0
    for s in range(n_super):
        super_u32[s] = running
        base = s * s_words
        end = min(n_words, base + s_words)

        within = 0
        for w in range(base, end):
            sub_u16[w] = within
            within += int(np.uint32(T_words_host_u32[w]).bit_count())
        running += within

    super_u32[n_super] = running
    return super_u32, sub_u16


def rank1_excl_cpu(
    T_words_u32: np.ndarray,
    super_u32: np.ndarray,
    sub_u16: np.ndarray,
    p: int,
    S_WORDS: int = DEFAULT_S_WORDS,
) -> int:
    word = p >> 5
    bit = p & 31
    s = word // S_WORDS
    base = int(super_u32[s])
    mid = int(sub_u16[word])
    if bit == 0:
        tail = 0
    else:
        mask = (1 << bit) - 1
        tail = int((int(T_words_u32[word]) & mask).bit_count())
    return base + mid + tail


def get_bit_cpu(words_u32: np.ndarray, bit_index: int) -> int:
    w = bit_index >> 5
    b = bit_index & 31
    return (int(words_u32[w]) >> b) & 1


def k2tree_adj_cpu(
    T_words_u32: np.ndarray,
    L_words_u32: np.ndarray,
    super_u32: np.ndarray,
    sub_u16: np.ndarray,
    u: int,
    v: int,
    N: int,
    H: int,
    Npad: int,
    T_nbits: int,
    k: int = DEFAULT_K,
    s_words: int = DEFAULT_S_WORDS,
) -> int:
    if not (0 <= u < N and 0 <= v < N):
        return 0
    if H == 0:
        return 0
    if H == 1:
        block = Npad // k  # ==1
        dr = u // block
        dc = v // block
        j = dr * k + dc
        return get_bit_cpu(L_words_u32, j)

    K2 = k * k
    base = 0
    size = Npad
    rank_incl = 0

    for level in range(H - 1):  # internal levels 0..H-2 in T
        block = size // k
        dr = u // block
        dc = v // block
        j = dr * k + dc
        q = base + j

        bit = get_bit_cpu(T_words_u32, q)
        if bit == 0:
            return 0

        rank_incl = rank1_excl_cpu(T_words_u32, super_u32, sub_u16, q, S_WORDS=s_words) + 1

        if level == H - 2:
            u = u % block
            v = v % block
            break

        base = K2 * rank_incl
        u = u % block
        v = v % block
        size = block

    leaf_base = K2 * rank_incl - T_nbits
    j_leaf = u * k + v
    return get_bit_cpu(L_words_u32, leaf_base + j_leaf)


def k2tree_trace_cpu(
    T_words_u32: np.ndarray,
    L_words_u32: np.ndarray,
    super_u32: np.ndarray,
    sub_u16: np.ndarray,
    u0: int,
    v0: int,
    N: int,
    H: int,
    Npad: int,
    T_nbits: int,
    k: int = DEFAULT_K,
    s_words: int = DEFAULT_S_WORDS,
) -> int:
    """
    Print the traversal path for a single (u,v), including final leaf node + leaf bit.
    Leaf bit is selected by descending to level H-2, then using j_leaf = (u%block)*k + (v%block).
    """
    print(f"TRACE u={u0} v={v0}  N={N} H={H} Npad={Npad} k={k}")

    if not (0 <= u0 < N and 0 <= v0 < N):
        print("  out of bounds => 0")
        return 0

    if H == 0:
        print("  H==0 => 0")
        return 0

    if H == 1:
        block = Npad // k  # ==1
        dr = u0 // block
        dc = v0 // block
        j = dr * k + dc
        bit = get_bit_cpu(L_words_u32, j)
        print(f"  H==1: block={block} dr={dr} dc={dc} j={j}  L[{j}]={bit}")
        return bit

    u = int(u0)
    v = int(v0)
    base = 0
    size = int(Npad)
    rank_incl = 0

    final_block = None
    final_u_in = None
    final_v_in = None

    for level in range(H - 1):  # internal levels 0..H-2
        block = size // k
        dr = u // block
        dc = v // block
        j = dr * k + dc
        q = base + j

        bit = get_bit_cpu(T_words_u32, q)
        rank_ex = rank1_excl_cpu(T_words_u32, super_u32, sub_u16, q, S_WORDS=s_words)
        rank_incl = rank_ex + 1

        print(
            f"  lvl={level:2d} size={size:6d} block={block:6d} "
            f"u={u:6d} v={v:6d} dr={dr} dc={dc} j={j:2d} q={q:8d} "
            f"T[q]={bit} rank_ex={rank_ex} rank_incl={rank_incl}"
        )

        if bit == 0:
            print("  stop: T[q]==0 => 0")
            return 0

        if level == H - 2:
            final_block = block
            final_u_in = u % block
            final_v_in = v % block
            break

        base = (k * k) * rank_incl
        u %= block
        v %= block
        size = block

    leaf_base = (k * k) * rank_incl - T_nbits
    j_leaf = int(final_u_in) * k + int(final_v_in)
    leaf_bit_index = int(leaf_base) + int(j_leaf)
    leaf_bit = get_bit_cpu(L_words_u32, leaf_bit_index)

    print(
        f"  leaf: rank_incl={rank_incl} leaf_base={leaf_base} "
        f"u_in={final_u_in} v_in={final_v_in} j_leaf={j_leaf} -> L[{leaf_bit_index}]={leaf_bit}"
    )
    return leaf_bit


def make_k2tree_kernels_u32(K_value: int = DEFAULT_K, s_words: int = DEFAULT_S_WORDS) -> "cp.RawKernel":
    """
    Compile adjacency kernel specialized for compile-time K.
    Layout:
      - T stores levels 0..H-2
      - L stores level H-1 (leaf), K2 bits per expanded node at level H-2
    """
    _require_cupy()
    K2_value = K_value * K_value
    src = rf"""
    #define K {K_value}
    #define K2 {K2_value}
    #define S_WORDS {s_words}

    extern "C" __device__ __forceinline__
    unsigned int get_bit_u32(const unsigned int* __restrict__ words, unsigned int bit_index)
    {{
        unsigned int w = bit_index >> 5;
        unsigned int b = bit_index & 31u;
        return (words[w] >> b) & 1u;
    }}

    extern "C" __device__ __forceinline__
    unsigned int rank1_excl_T_u32(const unsigned int* __restrict__ T_words,
                                 const unsigned int* __restrict__ super_u32,
                                 const unsigned short* __restrict__ sub_u16,
                                 unsigned int p)
    {{
        unsigned int word = p >> 5;
        unsigned int bit  = p & 31u;
        unsigned int s    = word / S_WORDS;

        unsigned int base = super_u32[s];
        unsigned int mid  = (unsigned int)sub_u16[word];

        unsigned int mask = (bit == 0u) ? 0u : ((1u << bit) - 1u);
        unsigned int tail = __popc(T_words[word] & mask);
        return base + mid + tail;
    }}

    extern "C" __global__
    void k2tree_adj_u32(const unsigned int* __restrict__ T_words,
                        const unsigned int* __restrict__ L_words,
                        const unsigned int* __restrict__ super_u32,
                        const unsigned short* __restrict__ sub_u16,
                        const int* __restrict__ u_i32,
                        const int* __restrict__ v_i32,
                        unsigned char* __restrict__ out_u8,
                        int n_queries,
                        int N, int H, int Npad, int T_nbits)
    {{
        for (int t = (int)(blockDim.x * blockIdx.x + threadIdx.x);
             t < n_queries;
             t += (int)(blockDim.x * gridDim.x))
        {{
            int u = u_i32[t];
            int v = v_i32[t];

            if ((unsigned int)u >= (unsigned int)N || (unsigned int)v >= (unsigned int)N) {{
                out_u8[t] = 0;
                continue;
            }}

            if (H == 0) {{
                out_u8[t] = 0;
                continue;
            }}

            if (H == 1) {{
                int size = Npad;
                int block = size / K; // ==1
                int dr = u / block;
                int dc = v / block;
                unsigned int j_leaf = (unsigned int)(dr * K + dc);
                out_u8[t] = (unsigned char)get_bit_u32(L_words, j_leaf);
                continue;
            }}

            unsigned int base = 0;
            int size = Npad;
            unsigned int rank_incl = 0;
            unsigned char hit = 1;

            for (int level = 0; level < H - 1; ++level)
            {{
                int block = size / K;
                int dr = u / block;
                int dc = v / block;
                unsigned int j = (unsigned int)(dr * K + dc);
                unsigned int q = base + j;

                unsigned int bit = get_bit_u32(T_words, q);
                if (bit == 0u) {{
                    hit = 0;
                    break;
                }}

                rank_incl = rank1_excl_T_u32(T_words, super_u32, sub_u16, q) + 1u;

                if (level == (H - 2)) {{
                    // Descend into the selected leaf node; then pick cell inside it from L.
                    u = u % block;
                    v = v % block;
                    break;
                }}

                base = (unsigned int)K2 * rank_incl;
                u = u % block;
                v = v % block;
                size = block;
            }}

            if (!hit) {{
                out_u8[t] = 0;
                continue;
            }}

            unsigned int leaf_base = (unsigned int)K2 * rank_incl - (unsigned int)T_nbits;
            unsigned int j_leaf = (unsigned int)(u * K + v);
            out_u8[t] = (unsigned char)get_bit_u32(L_words, leaf_base + j_leaf);
        }}
    }}
    """
    return cp.RawKernel(src, "k2tree_adj_u32")


def get_k2tree_kernel(K_value: int = DEFAULT_K, s_words: int = DEFAULT_S_WORDS) -> "cp.RawKernel":
    _require_cupy()
    key = (int(K_value), int(s_words))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = make_k2tree_kernels_u32(K_value=K_value, s_words=s_words)
        _KERNEL_CACHE[key] = kernel
    return kernel


@dataclass
class K2Tree:
    K: int
    S_WORDS: int
    N: int
    Npad: int
    H: int
    T_nbits: int
    T_words: np.ndarray | "cp.ndarray"
    L_words: np.ndarray | "cp.ndarray"
    super_u32: np.ndarray | "cp.ndarray"
    sub_u16: np.ndarray | "cp.ndarray"
    on_gpu: bool
    _kernel: "cp.RawKernel | None" = None

    @classmethod
    def _from_host_arrays(
        cls,
        *,
        T_words_h: np.ndarray,
        L_words_h: np.ndarray,
        super_h: np.ndarray,
        sub_h: np.ndarray,
        H: int,
        Npad: int,
        N_used: int,
        T_nbits: int,
        k: int,
        s_words: int,
        use_gpu: bool,
    ) -> "K2Tree":
        if use_gpu:
            cp_mod = _require_cupy()
            T_words_d = cp_mod.asarray(T_words_h, dtype=cp_mod.uint32)
            L_words_d = cp_mod.asarray(L_words_h, dtype=cp_mod.uint32)
            super_d = cp_mod.asarray(super_h, dtype=cp_mod.uint32)
            sub_d = cp_mod.asarray(sub_h, dtype=cp_mod.uint16)
            kernel = get_k2tree_kernel(K_value=k, s_words=s_words)
            return cls(
                K=int(k),
                S_WORDS=int(s_words),
                N=int(N_used),
                Npad=int(Npad),
                H=int(H),
                T_nbits=int(T_nbits),
                T_words=T_words_d,
                L_words=L_words_d,
                super_u32=super_d,
                sub_u16=sub_d,
                on_gpu=True,
                _kernel=kernel,
            )

        return cls(
            K=int(k),
            S_WORDS=int(s_words),
            N=int(N_used),
            Npad=int(Npad),
            H=int(H),
            T_nbits=int(T_nbits),
            T_words=T_words_h,
            L_words=L_words_h,
            super_u32=super_h,
            sub_u16=sub_h,
            on_gpu=False,
            _kernel=None,
        )

    @classmethod
    def from_adjdict(
        cls,
        adj: Dict[int, List[int]],
        N: int | None = None,
        k: int = DEFAULT_K,
        s_words: int = DEFAULT_S_WORDS,
        use_gpu: bool = True,
        verify_all: bool = True,
        verify_progress_every: int | None = 1000,
    ) -> "K2Tree":
        T_words_h, L_words_h, H, Npad, N_used, T_nbits = build_k2tree_from_adjdict(adj, N=N, k=k)
        super_h, sub_h = build_rank_tables_u32(T_words_h, s_words=s_words)
        if verify_all:
            _verify_adjdict_full(
                adj,
                T_words_h,
                L_words_h,
                super_h,
                sub_h,
                N_used,
                H,
                Npad,
                T_nbits,
                k,
                s_words,
                progress_every=verify_progress_every,
            )
        return cls._from_host_arrays(
            T_words_h=T_words_h,
            L_words_h=L_words_h,
            super_h=super_h,
            sub_h=sub_h,
            H=H,
            Npad=Npad,
            N_used=N_used,
            T_nbits=T_nbits,
            k=k,
            s_words=s_words,
            use_gpu=use_gpu,
        )

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Tuple[int, int]],
        N: int | None = None,
        k: int = DEFAULT_K,
        s_words: int = DEFAULT_S_WORDS,
        use_gpu: bool = True,
        verify_all: bool = False,
        verify_progress_every: int | None = 1000,
    ) -> "K2Tree":
        T_words_h, L_words_h, H, Npad, N_used, T_nbits = build_k2tree_from_edges(edges, N=N, k=k)
        super_h, sub_h = build_rank_tables_u32(T_words_h, s_words=s_words)
        if verify_all:
            adj: Dict[int, List[int]] = {}
            for u, v in edges:
                u_i = int(u)
                v_i = int(v)
                adj.setdefault(u_i, []).append(v_i)
            _verify_adjdict_full(
                adj,
                T_words_h,
                L_words_h,
                super_h,
                sub_h,
                N_used,
                H,
                Npad,
                T_nbits,
                k,
                s_words,
                progress_every=verify_progress_every,
            )
        return cls._from_host_arrays(
            T_words_h=T_words_h,
            L_words_h=L_words_h,
            super_h=super_h,
            sub_h=sub_h,
            H=H,
            Npad=Npad,
            N_used=N_used,
            T_nbits=T_nbits,
            k=k,
            s_words=s_words,
            use_gpu=use_gpu,
        )

    @classmethod
    def from_adjacency_matrix(
        cls,
        matrix: Any,
        N: int | None = None,
        k: int = DEFAULT_K,
        s_words: int = DEFAULT_S_WORDS,
        use_gpu: bool = True,
        verify_all: bool = False,
        verify_progress_every: int | None = 1000,
    ) -> "K2Tree":
        T_words_h, L_words_h, H, Npad, N_used, T_nbits = build_k2tree_from_adjacency_matrix(
            matrix, N=N, k=k
        )
        super_h, sub_h = build_rank_tables_u32(T_words_h, s_words=s_words)
        if verify_all:
            if hasattr(matrix, "tocoo"):
                coo = matrix.tocoo()
                edges = list(zip(coo.row.tolist(), coo.col.tolist()))
            else:
                dense = np.asarray(matrix)
                rows, cols = np.nonzero(dense)
                edges = list(zip(rows.tolist(), cols.tolist()))
            adj: Dict[int, List[int]] = {}
            for u, v in edges:
                adj.setdefault(int(u), []).append(int(v))
            _verify_adjdict_full(
                adj,
                T_words_h,
                L_words_h,
                super_h,
                sub_h,
                N_used,
                H,
                Npad,
                T_nbits,
                k,
                s_words,
                progress_every=verify_progress_every,
            )
        return cls._from_host_arrays(
            T_words_h=T_words_h,
            L_words_h=L_words_h,
            super_h=super_h,
            sub_h=sub_h,
            H=H,
            Npad=Npad,
            N_used=N_used,
            T_nbits=T_nbits,
            k=k,
            s_words=s_words,
            use_gpu=use_gpu,
        )

    @classmethod
    def load_npz(cls, path: str, use_gpu: bool = True) -> "K2Tree":
        with np.load(path) as data:
            k = int(data["K"]) if "K" in data else DEFAULT_K
            s_words = int(data["S_WORDS"]) if "S_WORDS" in data else DEFAULT_S_WORDS
            N_used = int(data["N"])
            Npad = int(data["Npad"])
            H = int(data["H"])
            T_nbits = int(data["T_nbits"])

            T_words_h = data["T_words"].astype(np.uint32, copy=False)
            L_words_h = data["L_words"].astype(np.uint32, copy=False)
            if "super_u32" in data:
                super_h = data["super_u32"].astype(np.uint32, copy=False)
            else:
                super_h = data["super"].astype(np.uint32, copy=False)
            if "sub_u16" in data:
                sub_h = data["sub_u16"].astype(np.uint16, copy=False)
            else:
                sub_h = data["sub"].astype(np.uint16, copy=False)

        return cls._from_host_arrays(
            T_words_h=T_words_h,
            L_words_h=L_words_h,
            super_h=super_h,
            sub_h=sub_h,
            H=H,
            Npad=Npad,
            N_used=N_used,
            T_nbits=T_nbits,
            k=k,
            s_words=s_words,
            use_gpu=use_gpu,
        )

    def __repr__(self) -> str:
        device = "gpu" if self.on_gpu else "cpu"
        return (
            f"K2Tree(K={self.K}, N={self.N}, H={self.H}, Npad={self.Npad}, "
            f"T_nbits={self.T_nbits}, device='{device}')"
        )

    def as_dict(self, host: bool = False) -> Dict[str, Any]:
        if host and self.on_gpu:
            cp_mod = _require_cupy()
            T_words = cp_mod.asnumpy(self.T_words)
            L_words = cp_mod.asnumpy(self.L_words)
            super_u32 = cp_mod.asnumpy(self.super_u32)
            sub_u16 = cp_mod.asnumpy(self.sub_u16)
        else:
            T_words = self.T_words
            L_words = self.L_words
            super_u32 = self.super_u32
            sub_u16 = self.sub_u16

        return {
            "T_words": T_words,
            "L_words": L_words,
            "super": super_u32,
            "sub": sub_u16,
            "H": int(self.H),
            "Npad": int(self.Npad),
            "N": int(self.N),
            "T_nbits": int(self.T_nbits),
            "K": int(self.K),
            "S_WORDS": int(self.S_WORDS),
        }

    def save_npz(self, path: str) -> None:
        data = self.as_dict(host=True)
        np.savez(
            path,
            K=np.int32(data["K"]),
            S_WORDS=np.int32(data["S_WORDS"]),
            N=np.int32(data["N"]),
            Npad=np.int32(data["Npad"]),
            H=np.int32(data["H"]),
            T_nbits=np.int64(data["T_nbits"]),
            T_words=data["T_words"],
            L_words=data["L_words"],
            super_u32=data["super"],
            sub_u16=data["sub"],
        )

    def to_cpu(self) -> "K2Tree":
        if not self.on_gpu:
            return self
        cp_mod = _require_cupy()
        return self._from_host_arrays(
            T_words_h=cp_mod.asnumpy(self.T_words),
            L_words_h=cp_mod.asnumpy(self.L_words),
            super_h=cp_mod.asnumpy(self.super_u32),
            sub_h=cp_mod.asnumpy(self.sub_u16),
            H=self.H,
            Npad=self.Npad,
            N_used=self.N,
            T_nbits=self.T_nbits,
            k=self.K,
            s_words=self.S_WORDS,
            use_gpu=False,
        )

    def to_gpu(self) -> "K2Tree":
        if self.on_gpu:
            return self
        return self._from_host_arrays(
            T_words_h=self.T_words,
            L_words_h=self.L_words,
            super_h=self.super_u32,
            sub_h=self.sub_u16,
            H=self.H,
            Npad=self.Npad,
            N_used=self.N,
            T_nbits=self.T_nbits,
            k=self.K,
            s_words=self.S_WORDS,
            use_gpu=True,
        )

    def recompile_kernel(self) -> None:
        if not self.on_gpu:
            raise RuntimeError("Cannot recompile kernel for a CPU-only K2Tree.")
        self._kernel = make_k2tree_kernels_u32(K_value=self.K, s_words=self.S_WORDS)

    def adj(self, u: int, v: int) -> int:
        if self.on_gpu:
            out = self.adj_batch(np.array([u], dtype=np.int32), np.array([v], dtype=np.int32))
            return int(out[0])
        return k2tree_adj_cpu(
            self.T_words,
            self.L_words,
            self.super_u32,
            self.sub_u16,
            int(u),
            int(v),
            self.N,
            self.H,
            self.Npad,
            self.T_nbits,
            k=self.K,
            s_words=self.S_WORDS,
        )

    def adj_batch(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.on_gpu:
            cp_mod = _require_cupy()
            u_d = cp_mod.asarray(u, dtype=cp_mod.int32)
            v_d = cp_mod.asarray(v, dtype=cp_mod.int32)
            if u_d.shape != v_d.shape:
                raise ValueError("u and v must have the same shape")
            out_shape = u_d.shape
            u_d = u_d.ravel()
            v_d = v_d.ravel()
            out_d = cp_mod.empty(u_d.size, dtype=cp_mod.uint8)

            threads = 256
            blocks = min(65535, (int(u_d.size) + threads - 1) // threads)

            kernel = self._kernel or get_k2tree_kernel(K_value=self.K, s_words=self.S_WORDS)
            kernel(
                (blocks,),
                (threads,),
                (
                    self.T_words,
                    self.L_words,
                    self.super_u32,
                    self.sub_u16,
                    u_d,
                    v_d,
                    out_d,
                    int(u_d.size),
                    int(self.N),
                    int(self.H),
                    int(self.Npad),
                    int(self.T_nbits),
                ),
            )
            return out_d.get().reshape(out_shape)

        u_arr = np.asarray(u, dtype=np.int32)
        v_arr = np.asarray(v, dtype=np.int32)
        if u_arr.shape != v_arr.shape:
            raise ValueError("u and v must have the same shape")

        flat_u = u_arr.ravel()
        flat_v = v_arr.ravel()
        out = np.fromiter(
            (
                k2tree_adj_cpu(
                    self.T_words,
                    self.L_words,
                    self.super_u32,
                    self.sub_u16,
                    int(uu),
                    int(vv),
                    self.N,
                    self.H,
                    self.Npad,
                    self.T_nbits,
                    k=self.K,
                    s_words=self.S_WORDS,
                )
                for uu, vv in zip(flat_u, flat_v)
            ),
            count=flat_u.size,
            dtype=np.uint8,
        )
        return out.reshape(u_arr.shape)

    def trace(self, u: int, v: int) -> int:
        if self.on_gpu:
            cp_mod = _require_cupy()
            T_h = cp_mod.asnumpy(self.T_words)
            L_h = cp_mod.asnumpy(self.L_words)
            super_h = cp_mod.asnumpy(self.super_u32)
            sub_h = cp_mod.asnumpy(self.sub_u16)
        else:
            T_h = self.T_words
            L_h = self.L_words
            super_h = self.super_u32
            sub_h = self.sub_u16
        return k2tree_trace_cpu(
            T_h,
            L_h,
            super_h,
            sub_h,
            int(u),
            int(v),
            self.N,
            self.H,
            self.Npad,
            self.T_nbits,
            k=self.K,
            s_words=self.S_WORDS,
        )


def k2tree_upload_from_adjdict(
    adj: Dict[int, List[int]],
    N: int | None = None,
    k: int = DEFAULT_K,
    s_words: int = DEFAULT_S_WORDS,
    use_gpu: bool = True,
    verify_all: bool = True,
    verify_progress_every: int | None = 1000,
) -> K2Tree:
    return K2Tree.from_adjdict(
        adj,
        N=N,
        k=k,
        s_words=s_words,
        use_gpu=use_gpu,
        verify_all=verify_all,
        verify_progress_every=verify_progress_every,
    )
