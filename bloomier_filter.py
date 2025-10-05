import numpy as np
from dataclasses import dataclass

@dataclass
class BloomierFilter:
    table: np.ndarray = None
    key_amount: int = -1
    neighb_count: int = -1

    def key_hasher(self, key, n):
        return ((key + n) * (key + n + 1)) // 2 + n

    def hash1(self, x):
        return (x * 2654435761) % self.key_amount

    def hash2(self, x):
        return (x * 1597334677) % self.key_amount

    def hash3(self, x):
        return (x * 3266489917) % self.key_amount

    def get_hashes(self, key):
        return set([self.hash1(key), self.hash2(key), self.hash3(key)])

    def find_peeling_order(self, keys):
        order = []
        remaining_keys = set(keys)
        cell_counts = [0] * self.key_amount  # how many unprocessed keys touch each cell
    
        # Initialize counts
        for key in keys:
            for cell in self.get_hashes(key):
                cell_counts[cell] += 1
    
        # Peel iteratively
        while remaining_keys:
            # Find a key with at least one cell that no other remaining key touches
            peelable_key = None
            unique_cell = None

            for key in remaining_keys:
                cells = self.get_hashes(key)
                for cell in cells:
                    if cell_counts[cell] == 1:  # only this key touches it
                        peelable_key = key
                        unique_cell = cell
                        break
                if peelable_key:
                    break
                
            if not peelable_key:
                return None  # construction failed - no valid peeling order

            #order.append((peelable_key, unique_cell))
            order.append(peelable_key)
            remaining_keys.remove(peelable_key)

            # Decrease counts for this key's cells
            for cell in self.get_hashes(peelable_key):
                cell_counts[cell] -= 1
    
        return list(reversed(order))

    def construct(self, neighbor_count, adj_list) -> bool:
        if neighbor_count != len(list(adj_list.items())[0][1]):
            return False

        self.neighb_count = neighbor_count
        self.key_amount = (neighbor_count * len(adj_list.keys())) * 3
        self.table = np.zeros((self.key_amount, ), dtype=np.int64)

        # process adj list into proper insertion format
        new_adj_list = {}
        debug_map = {}
        for key, value in adj_list.items():
            for i in range(0, neighbor_count):
                debug_map[(key, i)] = self.key_hasher(key, i)
                new_adj_list[self.key_hasher(key, i)] = value[i]
        adj_list = new_adj_list

        # get all hashes for each key
        taken_cells = {i: set([]) for i in range(self.key_amount+1)}
        for key, _ in adj_list.items():
            taken_cells[self.hash1(key)].add(key)
            taken_cells[self.hash2(key)].add(key)
            taken_cells[self.hash3(key)].add(key)

        peeling_order = self.find_peeling_order(adj_list.keys())

        # insert each key value pair in the correct peeling order
        # a key is peelable if:
        #   - it has at least one cell that is not assigned yet
        #     AND
        #   - isn't needed by another key 
        for key in peeling_order:
            value = adj_list[key]
            hashes = self.get_hashes(key)
            if any(self.table[h] == 0 and len(taken_cells[h]) == 1 for h in hashes):
                if all(self.table[h] != 0 for h in hashes):
                    #   - no cells open (something went wrong)
                    return False
                # insertion math: 
                # table[x] xor table[y] xor table[z] = v 
                # for k -> v where x = h1(k), y = h2(k), z = h3(k)

                #   - some or all cells open
                #       - Substitute assigned values into equation
                #       - Pick arbitrary values for all but one unassigned cell
                #       - Compute the remaining cell
                equation_values = [value]
                cell_to_solve_for = -1
                for h in hashes:
                    cellvalue = self.table[h]
                    if self.table[h] == 0:
                        if cell_to_solve_for == -1:
                            cell_to_solve_for = h
                            continue

                        cellvalue = np.random.randint(0, 255) 
                        self.table[h] = cellvalue
                    equation_values.append(cellvalue)

                self.table[cell_to_solve_for] = np.bitwise_xor.reduce(equation_values)
            else:
                print(f"failed to insert key: {key} with value: {value}")

        return True
                
    def get_neighbors(self, node):
        neighbors = []
        for i in range(0, self.neighb_count):
            hashes = self.get_hashes(self.key_hasher(node, i))
            cells = np.array([self.table[h] for h in hashes])
            neighbors.append(np.bitwise_xor.reduce(cells))
        return np.array(neighbors)



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
        
        
        
        
        
        
        
        





    
    