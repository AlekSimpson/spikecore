from weights_cuda import WeightMatrixCUDA
import matplotlib.animation as animation
import plotly.graph_objects as go, time
from IPython.display import display
from dataclasses import dataclass
import matplotlib.pyplot as plt
import threading, queue
import ipywidgets as w
from PIL import Image
from tqdm import tqdm
import cupy as cp
import warnings
import io
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
        learning_rate=0.00222):

        self.RESTING_MP = cp.float32(resting_mp)
        self.DECAY_RATE = cp.float32(decay_rate)
        self.LEARNING_RATE = cp.float32(learning_rate ) # 0.0033
        self.SPIKE_PERIOD = cp.int32(1)
        self.SPIKE_THRESHOLD = cp.float32(1)

        self.shape = shape
        self.neuron_count = self.shape[0] * self.shape[1]
        print("Constructing weight matrix...")
        self.weights = WeightMatrixCUDA(network, rank, weight_initializer)
        print("Weights constructed.")
        self.inputs = cp.zeros((self.neuron_count, ), dtype=cp.float32)
        self.membrane_potentials = cp.empty((self.neuron_count, ), dtype=cp.float32)
        self.membrane_potentials.fill(self.RESTING_MP)

        self.mp_logs = cp.zeros((self.neuron_count, 0), dtype=cp.float32)
        self.last_spiked = cp.zeros((self.neuron_count, ), dtype=cp.int32)

        self.alive = True

        self.threads = 1024
        self.blocks = (self.neuron_count + self.threads - 1) // self.threads

    def _setup_lifetime(self, lifetime: int):
        self.lifetime = lifetime
        if lifetime < 0:
            return

        self.mp_logs = cp.zeros((self.neuron_count, self.lifetime), dtype=cp.float32)

    def set_input_neurons(self, input_list: list): 
        if input_list == None:
            return
        self.input_neurons = cp.array(input_list)

    def start_static_record(self, input_spikes: cp.ndarray, lifetime: int, filename: str):
        step_src = open("cuda_code/kernels.c", "r").read();
        step_src = step_src.replace("<<NEIGHB_COUNT_SUB>>", str(self.weights.bloomier.neighb_count))
        step_src = step_src.replace("<<K_SUB>>", str(self.weights.U.shape[1]))
        step_kernel = cp.RawKernel(step_src, "step_kernel")

        self._setup_lifetime(lifetime)
        tick = 0
        if len(self.input_neurons) == 0:
            print("Set input neurons before starting the simulation.")
            return
        self.recording = True
        with tqdm(total=self.lifetime) as progress:
            with open(filename, "wb") as f:
                f.write(self.neuron_count.to_bytes(4, "big"))
                while tick < self.lifetime:
                    self.inputs[self.input_neurons] += input_spikes[tick]
                    self.step(tick, kernel=step_kernel)
                    # f.write(self.membrane_potentials.get().tobytes())
                    tick += 1
                    progress.update(1)
        self.recording = False
        print(f"Recording saved: {filename}")

    def step(self, tick, kernel):
        kernel(
            (self.blocks, ), (self.threads, ),
            (
                tick,
                self.SPIKE_PERIOD,
                self.SPIKE_THRESHOLD,
                self.LEARNING_RATE,
                self.DECAY_RATE,
                self.RESTING_MP,
                self.weights.bloomier.salt,
                cp.uint64(0xFFFFFFFFFFFFFFFF),
                self.weights.bloomier.key_amount,
                self.weights.U,
                self.weights.V,
                self.weights.bloomier.table,
                self.neuron_count,
                self.inputs,
                self.membrane_potentials,
                self.last_spiked
            )
        )













