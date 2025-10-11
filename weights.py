import numpy as np
from dataclasses import dataclass

@dataclass
class WeightMatrix:
    u: np.ndarray = None
    v: np.ndarray = None
    check: bool = True

    def get_element(self, i, j):
        return np.einsum('...k,...k->...', self.u[i], self.v[j])
        #return self.u[i] @ self.v[j].T

    def __getitem__(self, key):
        # Valid Indexing Check
        if self.check:
            if not all(isinstance(k, (int, list, np.int64, np.ndarray)) for k in key):
                raise TypeError("Each element must be int or list of ints")
            if any(isinstance(k, list) and not all(isinstance(i, int) for i in k) for k in key):
                raise TypeError("Lists must contain only ints")

        return self.get_element(*key)

    def __setitem__(*args):
        raise NotImplementedError("Setting values directly is unsupported.")

    def update_element_step(self, i, j, delta, lr, l2_reg, iters=1):
        # broadcast pairwise like NumPy fancy indexing
        I, J = np.broadcast_arrays(np.asarray(i), np.asarray(j))
        D = np.broadcast_to(np.asarray(delta), I.shape)

        # flatten pairs
        If = I.ravel()
        Jf = J.ravel()
        Df = D.ravel()

        # row/col anchors (only touched rows/cols, not full copies)
        Ui_unique, inv_i = np.unique(If, return_inverse=True)
        Vj_unique, inv_j = np.unique(Jf, return_inverse=True)
        U_anchor = self.u[Ui_unique].copy()
        V_anchor = self.v[Vj_unique].copy()

        for _ in range(iters):
            ui = self.u[If]                      # (P,k)
            vj = self.v[Jf]                      # (P,k)

            den_v = np.sum(vj**2, axis=1, keepdims=True) + l2_reg
            den_u = np.sum(ui**2, axis=1, keepdims=True) + l2_reg

            du = lr * (Df[:, None] * (vj / den_v) - l2_reg * (ui - U_anchor[inv_i]))
            dv = lr * (Df[:, None] * (ui / den_u) - l2_reg * (vj - V_anchor[inv_j]))

            # scatter-add back to rows/cols; handles repeats in I/J
            np.add.at(self.u, If, du)
            np.add.at(self.v, Jf, dv)


    def save(self, filepath):
        np.savez_compressed(filepath, u=self.u, v=self.v)
    def load_from_disk(self, filepath):
        try:
            data = np.load(filepath)
            self.u = data['u']
            self.v = data['v']
        except:
            raise Exception(f"Couldn't open file: {filepath}")



def create_weight_matrix(uncompressed_shape) -> WeightMatrix:
    pass
