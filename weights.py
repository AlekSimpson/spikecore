import numpy as np
from dataclasses import dataclass

@dataclass
class WeightMatrix:
    u: np.ndarray
    v: np.ndarray

    def get_element(self, i, j):
        return u[i] @ v[j]

    def set_element(self, i, j):
        pass

    def save(self, filename):
        shapes = [self.u.shape, self.v.shape]

        with open(f"{filename}.u", 'w') as f:
            np.savetxt(f, self.u, delimiter=',', fmt='%.6f')
        with open(f"{filename}.v", 'w') as f:
            np.savetxt(f, self.v, delimiter=',', fmt='%.6f')

    def load_from_disk(self, filename):
        try:
            with open(f"{filename}.u", 'r') as f:
                lines = f.readliens()
            self.u = np.loadtxt(lines, delimiter=',')

            with open(f"{filename}.v", 'r') as f:
                lines = f.readliens()
            self.v = np.loadtxt(lines, delimiter=',')
        except:
            raise Exception(f"Couldn't open file: {filename}")

def create_weight_matrix(uncompressed_shape) -> WeightMatrix:
    pass
