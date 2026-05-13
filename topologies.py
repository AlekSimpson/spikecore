def square_torus(k: int) -> dict[int, list[int]]:
    return {
        i: [
            (i//k)*k + (i%k + 1) % k,     # right
            (i//k)*k + (i%k - 1) % k,     # left
            ((i//k + 1) % k)*k + (i%k),   # down
            ((i//k - 1) % k)*k + (i%k),   # up
        ]
        for i in range(k*k)
    }


def small_world_torus(k: int, random_fanout: int = 4, seed: int | None = None) -> dict[int, list[int]]:
    """Torus neighbors plus fixed-count long-range shortcuts.

    The CUDA engine expects every neuron to have the same number of children.
    This preserves the four local torus edges and adds deterministic random
    shortcuts so activity can mix across the reservoir instead of only drifting
    locally.
    """
    import numpy as np

    if random_fanout < 0:
        raise ValueError("random_fanout must be non-negative.")

    base = square_torus(k)
    n = k * k
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        children = list(base[i])
        used = set(children)
        used.add(i)
        while len(children) < 4 + random_fanout:
            candidate = int(rng.integers(0, n))
            if candidate in used:
                continue
            used.add(candidate)
            children.append(candidate)
        out[i] = children
    return out


def random_fixed_outdegree(k: int, fanout: int = 8, seed: int | None = None) -> dict[int, list[int]]:
    """Directed random graph with the same out-degree for every neuron.

    The CUDA weight matrix requires a fixed child count per source neuron.
    This intentionally removes the torus/local-grid edges while keeping the
    reservoir size and input-neuron locations unchanged.
    """
    import numpy as np

    n = k * k
    fanout = int(fanout)
    if fanout <= 0:
        raise ValueError("fanout must be positive.")
    if fanout >= n:
        raise ValueError("fanout must be smaller than the neuron count.")

    rng = np.random.default_rng(seed)
    out = {}
    all_nodes = np.arange(n, dtype=np.int32)
    for i in range(n):
        candidates = all_nodes[all_nodes != i]
        out[i] = rng.choice(candidates, size=fanout, replace=False).astype(int).tolist()
    return out

