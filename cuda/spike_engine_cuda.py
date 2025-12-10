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

        self.RESTING_MP = resting_mp
        self.DECAY_RATE = decay_rate
        self.LEARNING_RATE = learning_rate # 0.0033
        self.SPIKE_PERIOD = 1
        self.SPIKE_THRESHOLD = 1

        self.shape = shape
        self.neuron_count = self.shape[0] * self.shape[1]
        print("Constructing weight matrix...")
        self.weights = WeightMatrixCUDA(network, rank, weight_initializer)
        print("Weights constructed.")
        self.neuron_inputs = cp.zeros((self.neuron_count, ))
        self.inputs = cp.zeros((self.neuron_count, ))
        self.membrane_potentials = cp.empty((self.neuron_count, ), dtype=cp.float32)
        self.membrane_potentials.fill(self.RESTING_MP)

        self.mp_logs = cp.zeros((self.neuron_count, 0), dtype=cp.float32)
        self.last_spiked = cp.zeros((self.neuron_count, ))

        self.alive = True

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
                    self.step(tick)
                    f.write(self.membrane_potentials.tobytes())
                    tick += 1
                    progress.update(1)
        self.recording = False
        print(f"Recording saved: {filename}")

    def step(self, tick):
        self.membrane_potentials += self.inputs

        self.inputs.fill(0)

        self.membrane_potentials[(tick - self.last_spiked) == self.SPIKE_PERIOD] = self.RESTING_MP 

        neurons_to_spike = cp.where(self.membrane_potentials > self.SPIKE_THRESHOLD)[0]
        self.spike(tick, neurons_to_spike)

        neurons_to_decay = cp.where(self.membrane_potentials <= self.SPIKE_THRESHOLD)[0]
        self.decay(neurons_to_decay)

    def spike(self, tick, neurons):
        last_spikes = self.last_spiked[neurons]
        expired = (tick - last_spikes) > self.SPIKE_PERIOD
        expired_neurons = neurons[expired]
        self.last_spiked[expired_neurons] = tick

        self.stdp(tick, neurons)

        children = self.weights.get_neighbors(neurons)
        self.inputs[children.ravel()] += self.weights[
            cp.broadcast_to(neurons[:, None], children.shape).ravel(), 
            children.ravel()
        ]

    def decay(self, neurons):
        self.membrane_potentials[neurons] += (self.RESTING_MP - self.membrane_potentials[neurons]) * self.DECAY_RATE

    def stdp(self, tick, neurons):
        children = self.weights.get_neighbors(neurons)
        do_hebb = ~((self.last_spiked[children] == 0) | (self.last_spiked[children] == tick))
        hebb_neurons = cp.broadcast_to(neurons[:, None], children.shape)[do_hebb]
        children = children[do_hebb]
        if cp.any(children):
            tick_delta = cp.abs(tick - self.last_spiked[children])
            decay_deltas = -self.LEARNING_RATE * tick_delta**-3
            self.weights.at[hebb_neurons, children.ravel()] <<= decay_deltas

    def rstdp(self, tick):
        pass













