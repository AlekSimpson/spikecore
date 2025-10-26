import numpy as np
from dataclasses import dataclass

MASK64: int = (1 << 64) - 1

@dataclass
class BloomierFilter:
    table: np.ndarray = None
    key_amount: int = -1
    neighb_count: int = -1
    salt: int = 0x243F6A8885A308D3  # or random 64-bit

    def key_hasher(self, key, n):
        return ((key + n) * (key + n + 1)) // 2 + n
    def _splitmix64(self, x):
        # add once in your class
        x = (x + 0x9E3779B97F4A7C15) & MASK64
        z = x
        z ^= (z >> 30)
        z = (z * 0xBF58476D1CE4E5B9) & MASK64
        z ^= (z >> 27)
        z = (z * 0x94D049BB133111EB) & MASK64
        z ^= (z >> 31)
        return z & MASK64

    def get_hashes(self, key):
        # robust, independent-ish indices; dedup to avoid double-decrement bugs
        x = (key ^ self.salt) & MASK64
        h1 = self._splitmix64(x) % self.key_amount
        h2 = self._splitmix64(x + 1) % self.key_amount
        h3 = self._splitmix64(x + 0x9D) % self.key_amount
        return h1, h2, h3

    # Replace find_peeling_order to RECORD the witness cell for each key and
    # to return the order already reversed for assignment.
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
        # Return in ASSIGNMENT order (reverse of peeling)
        return list(reversed(peel_order)), witness

    # Adjust reseed helper to new return signature
    def build_with_reseeds(self, keys, max_tries=16):
        for _ in range(max_tries):
            order, witness = self.find_peeling_order(keys)
            if order:
                return order, witness
            self.salt = self._splitmix64(self.salt + 0x9E3779B97F4A7C15)
        return [], {}

    # Replace construct’s insertion loop: drop taken_cells and zero-sentinel logic.
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

        # peel
        order, witness = self.find_peeling_order(adj_list.keys())
        if not order:
            return False

        assigned = np.zeros(self.key_amount, dtype=bool)

        # assign in reverse-peel order using recorded witness cell
        for k in order:
            v = int(adj_list[k])
            h1, h2, h3 = self.get_hashes(k)
            c_star = witness[k]  # the unique cell for this key during peel
            others = [h for h in (h1, h2, h3) if h != c_star]

            rhs = v
            for h in others:
                rhs ^= int(self.table[h])  # other cells already assigned by construction
            self.table[c_star] = rhs
            assigned[c_star] = True

        return True

    def get_neighbors(self, nodes):
        nodes = np.atleast_1d(nodes).astype(np.uint64)
        # n_nodes = len(nodes)
    
        # Vectorize Cantor pairing
        indices = np.arange(self.neighb_count, dtype=np.uint64)
        keys = ((nodes[:, None] + indices) * (nodes[:, None] + indices + np.uint64(1))) // np.uint64(2) + indices
    
        # Vectorize hashing
        x = (keys ^ np.uint64(self.salt)) & MASK64
        h1 = self._splitmix64(x) % np.uint64(self.key_amount)
        h2 = self._splitmix64(x + np.uint64(1)) %  np.uint64(self.key_amount) 
        h3 = self._splitmix64(x + np.uint64(0x9D)) % np.uint64(self.key_amount)
    
        h1_idx = h1.astype(np.int64)
        h2_idx = h2.astype(np.int64)
        h3_idx = h3.astype(np.int64)
        result = self.table[h1_idx] ^ self.table[h2_idx] ^ self.table[h3_idx]

        return result

    def child_iter(self, node):
        for i in range(0, self.neighb_count):
            hashes = self.get_hashes(self.key_hasher(node, i))
            cells = np.array([self.table[h] for h in hashes])
            yield np.bitwise_xor.reduce(cells)


# network = {
#     1: [3, 2],
#     2: [3, 4],
#     3: [2, 1],
#     4: [2, 3]
# }
# 
# bf = BloomierFilter()
# success = bf.construct(2, network)
# 
# if success:
#     print("bloomier filter created successfully")
# else:
#     print("failed to construct bloom filter")
# 
# 
# print(f"neighbors of 1: {bf.get_neighbors(1)} | actual: [3, 2]")
# print(f"neighbors of 2: {bf.get_neighbors(2)} | actual: [3, 4]")
# print(f"neighbors of 3: {bf.get_neighbors(3)} | actual: [2, 1]")
# print(f"neighbors of 4: {bf.get_neighbors(4)} | actual: [2, 3]")
        
        
        
        
        
        
        
        





    
    