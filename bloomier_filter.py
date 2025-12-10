import numpy as np
from dataclasses import dataclass

# Use full 64-bit mask (unsigned)
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

@dataclass
class BloomierFilter:
    table: np.ndarray = None
    key_amount: int = -1
    neighb_count: int = -1
    salt = np.uint64(0x243F6A8885A308D3)  # or random 64-bit

    def key_hasher(self, key, n):
        return ((key + n) * (key + n + 1)) // 2 + n

    def _splitmix64(self, x):
        # Normalize to unsigned 64-bit array (works for scalars and arrays)
        x = np.asarray(x, dtype=np.uint64)
        # SplitMix64 steps
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK64
        z = x.copy()
        z ^= (z >> np.uint64(30))
        z = (z * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
        z ^= (z >> np.uint64(27))
        z = (z * np.uint64(0x94D049BB133111EB)) & MASK64
        z ^= (z >> np.uint64(31))
        return z & MASK64

    def get_hashes(self, key):
        # Scalar hashing; return Python ints
        key = np.uint64(key)
        x = (key ^ self.salt) & MASK64
        h1 = int(self._splitmix64(x) % np.uint64(self.key_amount))
        h2 = int(self._splitmix64(x + np.uint64(1)) % np.uint64(self.key_amount))
        h3 = int(self._splitmix64(x + np.uint64(0x9D)) % np.uint64(self.key_amount))
        return h1, h2, h3

    def find_peeling_order(self, keys):
        key_cells = {}
        max_cell = -1
        for k in keys:
            cells = tuple(sorted(set(self.get_hashes(k))))
            key_cells[k] = cells
            if cells:
                max_cell = max(max_cell, max(cells))
        if max_cell < 0:
            return [], {}

        m = max_cell + 1
        cell_to_keys = [set() for _ in range(m)]
        for k, cells in key_cells.items():
            for c in cells:
                cell_to_keys[c].add(k)

        from collections import deque
        deg = [len(s) for s in cell_to_keys]
        q = deque([c for c, d in enumerate(deg) if d == 1])

        remaining = set(keys)
        peel_order = []
        witness = {}

        while q and remaining:
            c = q.popleft()
            if deg[c] != 1:
                continue
            (k,) = tuple(cell_to_keys[c])
            if k not in remaining:
                continue
            peel_order.append(k)
            witness[k] = c
            remaining.remove(k)
            for c2 in key_cells[k]:
                if k in cell_to_keys[c2]:
                    cell_to_keys[c2].remove(k)
                    deg[c2] -= 1
                    if deg[c2] == 1:
                        q.append(c2)

        if remaining:
            return [], {}
        return list(reversed(peel_order)), witness

    def build_with_reseeds(self, keys, max_tries=16):
        for _ in range(max_tries):
            order, witness = self.find_peeling_order(keys)
            if order:
                return order, witness
            # advance salt deterministically, keep dtype
            self.salt = self._splitmix64(self.salt + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
            # If result is array (shouldn’t be), force scalar
            if isinstance(self.salt, np.ndarray):
                self.salt = np.uint64(self.salt.item())
        return [], {}

    def check_topology_cached():
        
        
        pass

    def construct(self, neighbor_count, adj_list) -> bool:
        if neighbor_count != len(list(adj_list.items())[0][1]):
            return False

        self.neighb_count = neighbor_count
        self.key_amount = (neighbor_count * len(adj_list.keys())) * 3
        self.table = np.zeros((self.key_amount,), dtype=np.int64)

        # expand (node -> neighbors[0..neighbor_count-1]) into keyed pairs via Cantor pairing
        new_adj_list = {}
        for key, value in adj_list.items():
            for i in range(neighbor_count):
                new_adj_list[self.key_hasher(key, i)] = value[i]
        adj_list = new_adj_list

        order, witness = self.find_peeling_order(adj_list.keys())
        if not order:
            return False

        assigned = np.zeros(self.key_amount, dtype=bool)

        for k in order:
            v = int(adj_list[k])
            h1, h2, h3 = self.get_hashes(k)
            c_star = witness[k]
            others = [h for h in (h1, h2, h3) if h != c_star]

            rhs = v
            for h in others:
                rhs ^= int(self.table[h])
            self.table[c_star] = rhs
            assigned[c_star] = True

        return True

    def get_neighbors(self, nodes):
        nodes = np.atleast_1d(nodes).astype(np.int64)
        indices = np.arange(self.neighb_count, dtype=np.int64)
        keys = ((nodes[:, None] + indices) * (nodes[:, None] + indices + 1)) // 2 + indices
        # vectorized hashing in uint64
        x = (keys.astype(np.uint64) ^ self.salt) & MASK64
        h1 = (self._splitmix64(x) % np.uint64(self.key_amount)).astype(np.int64)
        h2 = (self._splitmix64(x + np.uint64(1)) % np.uint64(self.key_amount)).astype(np.int64)
        h3 = (self._splitmix64(x + np.uint64(0x9D)) % np.uint64(self.key_amount)).astype(np.int64)
        result = self.table[h1] ^ self.table[h2] ^ self.table[h3]
        return result if result.ndim > 1 else result.ravel()

    def child_iter(self, node):
        for i in range(0, self.neighb_count):
            hashes = self.get_hashes(self.key_hasher(node, i))
            cells = np.array([self.table[h] for h in hashes], dtype=np.int64)
            yield np.bitwise_xor.reduce(cells)