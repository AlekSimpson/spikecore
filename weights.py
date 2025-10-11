import numpy as np
from dataclasses import dataclass
from bloomier_filter import BloomierFilter

@dataclass
class WeightMatrix:
    """
    u: np.ndarray
    v: np.ndarray
    check: bool
    """

    _NO_STORE = object()

    def __init__(self, u: np.ndarray, v: np.ndarray, network: dict, check: bool = True):

        children_counts = {len(v) for v in network.values()}

        if not len(children_counts) == 1:
            raise TypeError("Inconsistent number of children for each neuron. Check your network dictionary.")

        self.bloomier = BloomierFilter()
        self.bloomier.construct(*children_counts, network)
        self.u = u
        self.v = v
        self.check = check


    @dataclass
    class _At:
        owner: "WeightMatrix"
        key: tuple | None = None

        def __getitem__(self, key):
            if self.key is not None:
                raise TypeError(
                    f"Chained indexing is invalid: .at[I,J][K,L], use only one index per .at"
                )

            self.key = self.owner.__checkkey__(key)
            return self

        def __setitem__(self, key, value):
            if value is not WeightMatrix._NO_STORE:
                raise NotImplementedError("Use '<<=' via .at, direct setting is unsupported.")
            self.key = None  # clear after augmented-assignment write-back

        def __ilshift__(self, delta):
            if self.key is None:
                raise TypeError("The .at interface must be indexed!")
            i, j = self.key
            self.owner.update(i, j, delta)
            return WeightMatrix._NO_STORE

    @property
    def at(self):
        return WeightMatrix._At(self)

    @dataclass
    class _Children:
        owner: "WeightMatrix"
        key: int | None = None

        def __getitem__(self, key):
            if self.key is not None:
                raise TypeError("Chained indexing like .children[a][b] is invalid")

            if isinstance(key, (int, np.integer)):
                self.key = key
            else:
                raise TypeError(f"Key must be an integer, but is a {type(key)}")
            return self

        def __iter__(self):
            if self.key is None:
                raise TypeError("The .children interface must be indexed.")
            yield from self.owner.bloomier.child_iter(self.key)
    @property
    def children(self):
        return WeightMatrix._Children(self)

    def __checkkey__(self,key):
        if self.check:
            if not all(isinstance(k, (int, list, np.int64, np.ndarray)) for k in key):
                raise TypeError("Each element must be int or list of ints")
            if any(isinstance(k, list) and not all(isinstance(i, int) for i in k) for k in key):
                raise TypeError("Lists must contain only ints")
        try:
            i, j = key
        except:
            raise TypeError(f"Key {key} is not a 2-tuple")

        return i,j
    def __getitem__(self, key):
        i,j = self.__checkkey__(key)
        return np.einsum('...k,...k->...', self.u[i], self.v[j])

    def __setitem__(*args):
        raise NotImplementedError("Setting values directly is unsupported.")

    def update(self, i, j, delta, lr=0.5, l2_reg=1, iters=1):
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
