from collections import defaultdict
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from tqdm import tqdm
import cupy as cp
import hashlib
import json
import time
import numpy as np

# Use full 64-bit mask (unsigned)
MASK64 = cp.uint64(0xFFFFFFFFFFFFFFFF)

@dataclass
class BloomierFilterCUDA:
    table: cp.ndarray = None
    key_amount: int = -1
    neighb_count: int = -1
    salt = cp.uint64(0x243F6A8885A308D3)  # or random 64-bit

    def key_hasher(self, key, n):
        return ((key + n) * (key + n + 1)) // 2 + n

    def _splitmix64(self, x):
        # Normalize to unsigned 64-bit array (works for scalars and arrays)
        x = cp.asarray(x, dtype=cp.uint64)
        # SplitMix64 steps
        x = (x + cp.uint64(0x9E3779B97F4A7C15)) & MASK64
        z = x.copy()
        z ^= (z >> cp.uint64(30))
        z = (z * cp.uint64(0xBF58476D1CE4E5B9)) & MASK64
        z ^= (z >> cp.uint64(27))
        z = (z * cp.uint64(0x94D049BB133111EB)) & MASK64
        z ^= (z >> cp.uint64(31))
        return z & MASK64

    def get_hashes(self, key):
        key = cp.uint64(key)
        x = (key ^ self.salt) & MASK64
        h = cp.array((3, ), dtype=cp.uint64)
        h.fill(x)
        h = h + cp.array([cp.uint64(0), cp.uint64(1), cp.uint64(0x9D)], dtype=cp.uint64)

        # split mix64 steps
        h = (h + cp.uint64(0x9E3779B97F4A7C15)) & MASK64
        # z = h.copy()
        h ^= (h >> cp.uint64(30))
        h = (h * cp.uint64(0xBF58476D1CE4E5B9)) & MASK64
        h ^= (h >> cp.uint64(27))
        h = (h * cp.uint64(0x94D049BB133111EB)) & MASK64
        h ^= (h >> cp.uint64(31))
        h = h & MASK64
        h = h % cp.uint64(self.key_amount)
        return h.get().tolist()

    def find_peeling_order(self, keys):
        keycells_matrix = cp.zeros((len(keys), 4), dtype=cp.uint64)
        keycells_matrix[:, 0] += cp.array(keys, dtype=cp.uint64) 
        keycells_matrix[:, 1] += cp.array(keys, dtype=cp.uint64) 
        keycells_matrix[:, 2] += cp.array(keys, dtype=cp.uint64) 
        keycells_matrix[:, 3] += cp.array(keys, dtype=cp.uint64) 
        
        keycells_matrix[:, 1:] = (keycells_matrix[:, 1:] ^ self.salt) & MASK64
        keycells_matrix[:, 2] += cp.uint64(1) 
        keycells_matrix[:, 3] += cp.uint64(0x9D)

        keycells_matrix[:, 1:] += cp.uint64(0x9E3779B97F4A7C15)
        keycells_matrix[:, 1:] ^= (keycells_matrix[:, 1:] >> cp.uint64(30))
        keycells_matrix[:, 1:] = (keycells_matrix[:, 1:] * cp.uint64(0xBF58476D1CE4E5B9)) & MASK64
        keycells_matrix[:, 1:] ^= (keycells_matrix[:, 1:] >> cp.uint64(27))
        keycells_matrix[:, 1:] = (keycells_matrix[:, 1:] * cp.uint64(0x94D049BB133111EB)) & MASK64
        keycells_matrix[:, 1:] ^= (keycells_matrix[:, 1:] >> cp.uint64(31))
        keycells_matrix[:, 1:] = keycells_matrix[:, 1:] & MASK64
        keycells_matrix[:, 1:] = keycells_matrix[:, 1:] % cp.uint64(self.key_amount)

        max_cell = keycells_matrix.max().get()
        key_cells = {}
        for i, k in tqdm(enumerate(keys), total=len(keys), desc="Building key cells"):
            key_cells[k] = keycells_matrix[i, 1:].get().tolist()
        if max_cell < 0:
            return [], {}, {}

        m = max_cell + 1
        cell_to_keys = [None] * m
        for k, cells in key_cells.items():
            for c in cells:
                if cell_to_keys[c] is None:
                    cell_to_keys[c] = set()
                cell_to_keys[c].add(k)
        deg = [len(s) if s is not None else 0 for s in cell_to_keys]
        q = deque([c for c, d in enumerate(deg) if d == 1])
        
        remaining = set(keys)
        peel_order = []
        witness = {}
        max_iterations = max(len(q), len(remaining))

        with tqdm(total=max_iterations, desc="Creating peeling order") as pbar:
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
                pbar.update(1)
                        
        if len(remaining) != 0:
            print("peeling failed")
            return [], {}, {}
            
        return list(reversed(peel_order)), witness, key_cells

    def build_with_reseeds(self, keys, max_tries=16):
        for _ in range(max_tries):
            order, witness = self.find_peeling_order(keys)
            if order:
                return order, witness
            # advance salt deterministically, keep dtype
            self.salt = self._splitmix64(self.salt + cp.uint64(0x9E3779B97F4A7C15)).astype(cp.uint64)
            # If result is array (shouldn’t be), force scalar
            if isinstance(self.salt, cp.ndarray):
                self.salt = cp.uint64(self.salt.item())
        return [], {}

    def hash_topology(self, obj):
       s = str(obj).encode()
       return hashlib.sha256(s).hexdigest()       

    def check_topology_cached(self, topology_hash):
        list_is_cached = Path("./.spikecore.cache/hashes.json").exists()

        # if running for first time, auto-create cache directory if it does not exist
        Path("./.spikecore.cache/").mkdir(parents=True, exist_ok=True)
        Path("./.spikecore.cache/hashes.json").touch(exist_ok=True)
        
        if not list_is_cached:
            return False

        try:
            with open("./.spikecore.cache/hashes.json", "r") as file:
                cache_data = json.load(file)
            return topology_hash in cache_data.keys()
        except:
            print("list not cached")

    def construct(self, neighbor_count, adj_list) -> bool:
        list_hash = self.hash_topology(adj_list)
        data = {}
        if self.check_topology_cached(list_hash):
            print("Weight topology cache detected. Loading from cache.")
            with open("./.spikecore.cache/hashes.json", "r") as file:
                data = json.load(file)
                saved_params = data[list_hash] 
                self.table = cp.array(saved_params[0])
                self.key_amount = saved_params[1]
                self.neighb_count = saved_params[2]
                self.salt = saved_params[3]
                
            return True
        
        if neighbor_count != len(list(adj_list.items())[0][1]):
            return False

        self.neighb_count = neighbor_count
        self.key_amount = (neighbor_count * len(adj_list.keys())) * 3
        self.table = cp.zeros((self.key_amount,), dtype=cp.int64)

        # expand (node -> neighbors[0..neighbor_count-1]) into keyed pairs via Cantor pairing
        new_adj_list = {}
        for key, value in adj_list.items():
            for i in range(neighbor_count):
                new_adj_list[self.key_hasher(key, i)] = value[i]
        adj_list = new_adj_list

        print("Finding peeling order...")
        start = time.perf_counter()
        order, witness, key_cells = self.find_peeling_order(list(adj_list.keys()))
        end = time.perf_counter()
        print(f"Done in {end - start:.4f} seconds.")
        if not order:
            return False

        assigned = cp.zeros(self.key_amount, dtype=bool)

        for k in tqdm(order, desc="Constructing Bloomier Filter"):
            v = int(adj_list[k])
            H = key_cells[k]
            c_star = witness[k]
            others = np.array([h for h in H if h != c_star])

            rhs = v
            for h in others:
                rhs ^= int(self.table[h])
            
            self.table[c_star] = rhs
            assigned[c_star] = True

        data[list_hash] = (self.table.get().tolist(), self.key_amount, self.neighb_count, int(self.salt))
        with open("./.spikecore.cache/hashes.json", "w") as file:
            json.dump(data, file, indent=2)
            
        return True

    def get_neighbors(self, nodes):
        nodes = cp.atleast_1d(nodes).astype(cp.int64)
        indices = cp.arange(self.neighb_count, dtype=cp.int64)
        keys = ((nodes[:, None] + indices) * (nodes[:, None] + indices + 1)) // 2 + indices

        
        x = (keys.astype(cp.uint64) ^ self.salt) & MASK64
        h1 = (self._splitmix64(x) % cp.uint64(self.key_amount)).astype(cp.int64)
        h2 = (self._splitmix64(x + cp.uint64(1)) % cp.uint64(self.key_amount)).astype(cp.int64)
        h3 = (self._splitmix64(x + cp.uint64(0x9D)) % cp.uint64(self.key_amount)).astype(cp.int64)
        result = self.table[h1] ^ self.table[h2] ^ self.table[h3]
        return result if result.ndim > 1 else result.ravel()

    def child_iter(self, node):
        for i in range(0, self.neighb_count):
            hashes = self.get_hashes(self.key_hasher(node, i))
            cells = cp.array([self.table[h] for h in hashes], dtype=cp.int64)
            yield cp.bitwise_xor.reduce(cells)