import cupy as cp
from dataclasses import dataclass
from bloomier_filter import BloomierFilter

@dataclass
class WeightMatrixCUDA:
    _NO_STORE = object()

    def __init__(self, network: dict, rank: int = None, check_indexing: bool = True, weight_initializer:callable = cp.random.normal):
        children_counts = {len(c) for c in network.values()}
        if len(children_counts) == 0:
            raise TypeError("?????")
        if not len(children_counts) == 1:
            raise TypeError("Inconsistent number of children for each neuron. Check your network dictionary.")

        n = len(network.keys())

        if rank is None:
            rank = cp.ceil(0.05 * n).astype(cp.int64)

        k = rank
        U,V = weight_initializer(size=(2, n, k))

        self.bloomier = BloomierFilter()
        self.bloomier.construct(*children_counts, network)
        # self.network = network
        self.U = U
        self.V = V
        self.check = check_indexing
        self.size = n

    def get_neighbors(self, i):
        return self.bloomier.get_neighbors(i)

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

            if isinstance(key, (int, cp.integer)):
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
            if not all(isinstance(k, (int, list, cp.int64, cp.ndarray)) for k in key):
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
        return cp.einsum('...k,...k->...', self.U[i], self.V[j])

    def __setitem__(*args):
        raise NotImplementedError("Setting values directly is unsupported.")

    def update(self, i, j, delta, lr=0.5, l2_reg=1, iters=1):
        # broadcast pairwise like NumPy fancy indexing
        I, J = cp.broadcast_arrays(cp.asarray(i), cp.asarray(j))
        D = cp.broadcast_to(cp.asarray(delta), I.shape)

        # flatten pairs
        If = I.ravel()
        Jf = J.ravel()
        Df = D.ravel()

        # row/col anchors (only touched rows/cols, not full copies)
        Ui_unique, inv_i = cp.unique(If, return_inverse=True)
        Vj_unique, inv_j = cp.unique(Jf, return_inverse=True)
        U_anchor = self.U[Ui_unique].copy()
        V_anchor = self.V[Vj_unique].copy()

        for _ in range(iters):
            ui = self.U[If]                      # (P,k)
            vj = self.V[Jf]                      # (P,k)

            den_v = cp.sum(vj**2, axis=1, keepdims=True) + l2_reg
            den_u = cp.sum(ui**2, axis=1, keepdims=True) + l2_reg

            du = lr * (Df[:, None] * (vj / den_v) - l2_reg * (ui - U_anchor[inv_i]))
            dv = lr * (Df[:, None] * (ui / den_u) - l2_reg * (vj - V_anchor[inv_j]))

            # scatter-add back to rows/cols; handles repeats in I/J
            cp.add.at(self.U, If, du)
            cp.add.at(self.V, Jf, dv)

    def save(self, filepath):
        cp.savez_compressed(filepath, u=self.U, v=self.V)

    def load_from_disk(self, filepath):
        try:
            data = cp.load(filepath)
            self.U = data['u']
            self.V = data['v']
        except:
            raise Exception(f"Couldn't open file: {filepath}")



def create_weight_matrix(uncompressed_shape) -> WeightMatrix:
    pass
