from weights_cuda import WeightMatrixCUDA
from dataclasses import dataclass
from tqdm import tqdm
import cupy as cp
import os
os.environ['CUPY_DUMP_CUDA_SOURCE_ON_ERROR'] = '1'


@dataclass
class SpikeEngineCUDA:
    neuron_count: int
    membrane_potentials: cp.ndarray
    weights: WeightMatrixCUDA
    RESTING_MP: float
    DECAY_RATE: float
    LEARNING_RATE: float
    SPIKE_PERIOD: int
    spike_threshold: int
    lifetime: int
    input_neurons: cp.ndarray
    neuron_inputs: cp.ndarray
    last_tick_updated: cp.ndarray
    live_input_vector: cp.ndarray
    alive: bool

    def __init__(
        self, 
        network: dict, 
        shape: tuple,
        rank: int = None, 
        weight_initializer: callable = cp.random.normal, 
        resting_mp=0.1,
        decay_rate=0.01,
        learning_rate=0.00222,
        use_k2tree: bool = True,
        verify_k2tree: bool = False,
        verify_progress_every: int | None = 1000,
    ):

        self.RESTING_MP = cp.float32(resting_mp)
        self.DECAY_RATE = cp.float32(decay_rate)
        self.LEARNING_RATE = cp.float32(learning_rate ) # 0.0033
        self.SPIKE_PERIOD = cp.int32(1)
        self.SPIKE_THRESHOLD = cp.float32(1)

        self.shape = shape
        self.neuron_count = self.shape[0] * self.shape[1]
        print("Constructing weight matrix...")
        self.weights = WeightMatrixCUDA(
            network,
            rank=rank,
            weight_initializer=weight_initializer,
            use_k2tree=use_k2tree,
            verify_k2tree=verify_k2tree,
            verify_progress_every=verify_progress_every,
        )
        print("Weights constructed.")
        self.inputs = cp.zeros((self.neuron_count, ), dtype=cp.float32)
        self.membrane_potentials = cp.empty((self.neuron_count, ), dtype=cp.float32)
        self.membrane_potentials.fill(self.RESTING_MP)

        self.mp_logs = cp.zeros((self.neuron_count, 0), dtype=cp.float32)
        self.last_spiked = cp.zeros((self.neuron_count, ), dtype=cp.int32)
        self.last_updated = cp.zeros((self.neuron_count, ), dtype=cp.int32)
        self.active = cp.empty((self.neuron_count, ), dtype=cp.int32)
        self.next_active = cp.empty((self.neuron_count, ), dtype=cp.int32)
        self.active_count = cp.zeros((1,), dtype=cp.int32)
        self.next_count = cp.zeros((1,), dtype=cp.int32)
        self.active_gen = cp.full((self.neuron_count, ), -1, dtype=cp.int32)
        self.step_kernel = None
        self.add_active_kernel = None

        self.alive = True

        self.threads = 256
        self.blocks = (self.neuron_count + self.threads - 1) // self.threads

    def _setup_lifetime(self, lifetime: int):
        self.lifetime = lifetime
        if lifetime < 0:
            return

        self.mp_logs = cp.zeros((self.neuron_count, self.lifetime), dtype=cp.float32)

    def _compile_kernels(self):
        step_src = open("cuda_code/kernels.c", "r").read();
        step_src = step_src.replace("<<NEIGHB_COUNT_SUB>>", str(self.weights.neighb_count))
        step_src = step_src.replace("<<K_SUB>>", str(self.weights.U.shape[1]))
        self.step_kernel = cp.RawKernel(step_src, "step_kernel")
        self.add_active_kernel = cp.RawKernel(step_src, "add_active_kernel")

    def _add_active(self, indices: cp.ndarray, tick: int):
        if indices is None or indices.size == 0:
            return
        if self.add_active_kernel is None:
            self._compile_kernels()
        idx = cp.asarray(indices, dtype=cp.int32).ravel()
        threads = 256
        blocks = (idx.size + threads - 1) // threads
        if blocks == 0:
            return
        self.add_active_kernel(
            (blocks,), (threads,),
            (
                idx,
                cp.int32(idx.size),
                cp.int32(tick),
                self.active,
                self.active_count,
                self.active_gen,
            ),
        )

    def set_input_neurons(self, input_list: list): 
        if input_list == None:
            return
        self.input_neurons = cp.asarray(input_list, dtype=cp.int32)

    def start_static_record(
        self,
        input_spikes: cp.ndarray,
        lifetime: int,
        filename: str,
        record_membrane: bool = True,
    ):
        if self.step_kernel is None:
            self._compile_kernels()
        self._setup_lifetime(lifetime)
        tick = 0
        input_neurons = getattr(self, "input_neurons", None)
        if input_neurons is None or len(input_neurons) == 0:
            print("Set input neurons before starting the simulation.")
            return
        input_spikes = cp.asarray(input_spikes, dtype=cp.float32)
        self.recording = True
        with tqdm(total=self.lifetime) as progress:
            with open(filename, "wb") as f:
                f.write(self.neuron_count.to_bytes(4, "big"))
                while tick < self.lifetime:
                    self.inputs[self.input_neurons] += input_spikes[tick]
                    self.next_count.fill(0)
                    self._add_active(self.input_neurons, tick)
                    self.step(tick)
                    if record_membrane:
                        f.write(self.membrane_potentials.get().tobytes())
                    self.active, self.next_active = self.next_active, self.active
                    self.active_count, self.next_count = self.next_count, self.active_count
                    # f.write(self.membrane_potentials.get().tobytes())
                    tick += 1
                    progress.update(1)
        self.recording = False
        print(f"Recording saved: {filename}")

    def estimate_bifurcation_weight(self, input_period: int = 2) -> tuple[float, float]:
        """
        Estimate per-spike weight thresholds for propagation.
        Returns (w_accum, w_instant).
        - w_accum: minimal constant input (per active tick) to eventually cross threshold.
        - w_instant: input needed to cross threshold in a single tick.
        """
        decay = float(self.DECAY_RATE)
        resting = float(self.RESTING_MP)
        threshold = float(self.SPIKE_THRESHOLD)
        decay_factor = (1.0 - decay) ** float(input_period)
        w_accum = (threshold - resting) * (1.0 - decay_factor)
        w_instant = (threshold - resting)
        return w_accum, w_instant

    def set_constant_weights_near_bifurcation(
        self,
        input_period: int = 2,
        scale: float = 1.2,
    ) -> tuple[float, float, float]:
        w_accum, w_instant = self.estimate_bifurcation_weight(input_period=input_period)
        target = w_accum * float(scale)
        self.weights.set_constant_weight(target)
        return target, w_accum, w_instant

    def step(self, tick: int):
        if self.step_kernel is None:
            self._compile_kernels()
        active_count = int(self.active_count.get())
        if active_count == 0:
            return
        threads = 256
        blocks = (active_count + threads - 1) // threads
        self.step_kernel(
            (blocks, ), (threads, ),
            (
                cp.int32(tick),
                cp.int32(tick + 1),
                self.SPIKE_PERIOD,
                self.SPIKE_THRESHOLD,
                self.LEARNING_RATE,
                self.DECAY_RATE,
                self.RESTING_MP,
                self.weights.U,
                self.weights.V,
                self.weights.neighbors,
                cp.int32(self.neuron_count),
                self.inputs,
                self.membrane_potentials,
                self.last_spiked,
                self.last_updated,
                self.active,
                self.active_count,
                self.next_active,
                self.next_count,
                self.active_gen,
            )
        )






