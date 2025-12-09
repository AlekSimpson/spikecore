from weights import WeightMatrix
import threading, queue
import ipywidgets as w
import plotly.graph_objects as go, time
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io
from tqdm import tqdm
from dataclasses import dataclass
import warnings

@dataclass 
class SpikeEngine:
    neuron_count: int
    membrane_potentials: np.ndarray
    weights: WeightMatrix
    RESTING_MP: float
    DECAY_RATE: float
    LEARNING_RATE: float
    SPIKE_PERIOD: int
    spike_threshold: int
    lifetime: int
    input_neurons: np.ndarray
    neuron_inputs: np.ndarray
    last_tick_updated: np.ndarray
    live_input_vector: np.ndarray
    alive: bool
    import numpy as np

    def __init__(
        self, 
        network: dict, 
        shape: tuple,
        rank: int = None, 
        weight_initializer: callable = np.random.normal, 
        resting_mp=0.1,
        decay_rate=0.01,
        learning_rate=0.00222):
        from pynput import keyboard

        self.RESTING_MP = resting_mp
        self.DECAY_RATE = decay_rate
        self.LEARNING_RATE = learning_rate # 0.0033
        self.SPIKE_PERIOD = 1
        self.SPIKE_THRESHOLD = 1

        self.shape = shape
        self.neuron_count = self.shape[0] * self.shape[1]
        print("Constructing weight matrix...")
        self.weights = WeightMatrix(network, rank=rank, weight_initializer=weight_initializer)
        print("Weights constructed.")
        self.neuron_inputs = np.zeros((self.neuron_count, ))
        self.inputs = np.zeros((self.neuron_count, ))
        self.membrane_potentials = np.empty((self.neuron_count, ), dtype=np.float32)
        self.membrane_potentials.fill(self.RESTING_MP)

        self.mp_logs = np.zeros((self.neuron_count, 0), dtype=np.float32)
        self.last_spiked = np.zeros((self.neuron_count, ))
        self.keybinds = {}

        self.alive = True

        # initial setup
        self.fig = go.FigureWidget()
        self.fig.add_trace(go.Heatmap(
            z=np.zeros(self.shape),
            colorscale="Viridis",
            zmin=0,
            zmax=10,

            # Performance optimizations
            showscale=False,       # No colorbar = faster
            hoverongaps=False,
            hoverinfo='skip',      # No tooltips = faster
            zauto=False,           # Fixed color scale = faster

            # WebGL-specific optimizations
            dx=1,  # Pixel spacing (optional)
            dy=1,
        ))
        self.fig.update_layout(
            width=500,
            height=500,
            margin=dict(l=0, r=0, b=0, t=0),
        
            # Hide axes for speed
            xaxis=dict(
                visible=False,
                fixedrange=True  # Disable zoom/pan for speed
            ),
            yaxis=dict(
                visible=False,
                fixedrange=True
            ),
        
            # Transparent backgrounds
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        
            # Critical: preserve UI state without recalculation
            uirevision='constant',
        
            # Disable modebar for cleaner look and slight speedup
            modebar=dict(remove=[
                'zoom', 
                'pan', 
                'select', 
                'lasso2d', 
                'zoomIn', 
                'zoomOut', 
                'autoScale', 
                'resetScale']
            ),
        )

        self.viz_buffer = np.zeros(self.shape, dtype=np.float32)

        # async display thread
        self.viz_q = queue.Queue(maxsize=2)
        self.stop_flag = threading.Event()

        # double-buffer latest frame (lock-protected)
        self.latest_frame = {"data": None, "t": -float("inf")}
        self.latest_lock = threading.Lock()

    def _setup_lifetime(self, lifetime: int):
        self.lifetime = lifetime
        if lifetime < 0:
            return

        self.mp_logs = np.zeros((self.neuron_count, self.lifetime), dtype=np.float32)

    def viz_loop(self):
        last_drawn_t = -float("inf")
        while not self.stop_flag.is_set():
            try:
                item = self.viz_q.get(timeout=0.02)
                # Consume to newest frame
                while True:
                    try:
                        item = self.viz_q.get_nowait()
                    except queue.Empty:
                        break
                    
                mP_grid, cur_t = item

                if cur_t > last_drawn_t:
                    # Single update, no intermediate copies
                    with self.fig.batch_update():
                        self.fig.data[0].z = mP_grid  # WebGL uploads to GPU here
                    last_drawn_t = cur_t

            except queue.Empty as e:
                pass
        
            time.sleep(0.01)

    def start_visual_loop(self):
        self.viz_thread = threading.Thread(
            target=self.viz_loop, 
            daemon=True, 
            name="viz_thread"
        )
        self.viz_thread.start()
        display(self.fig)

    def set_input_neurons(self, input_list: list): 
        if input_list == None:
            return
        self.input_neurons = np.array(input_list)

    def set_live_input_keybindings(self, bindings: dict):
        """
        bindings: keybind char -> input neuron id
        """
        if not self.input_neurons:
            print("Cannot set live input keybinds if no input neurons are set")
            return
        
        self.keybinds = bindings

    def start_static_record(self, input_spikes: np.ndarray, lifetime: int, filename: str):
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

    def on_press(self, key):
        try:
            if key.char == 'q':
                self.alive = False

            if key.char in self.keybinds:
                self.inputs[self.keybinds[key.char]] += 1
            print(f'Key pressed: {key.char}')
        except AttributeError:
            print(f'Special key pressed: {key}')

    def on_release(self, key):
        pass

    def start_dynamic(self, granularity=1, speed=0):
        # live, undetermined dynamic network inputs, undetermined simulation lifetime
        if speed < 0:
            print("'speed' parameter only slows simulation down. Cannot be negative.")
            return 
        if granularity < 0:
            print("granularity must be greater than 0.")
            return

        self._setup_lifetime(-1)
        tick = 0

        if len(self.input_neurons) == 0:
            print("Set input neurons before starting the simulation.")
            return

        listener = keyboard.Listener(
            on_press=self.on_press, 
            on_release=self.on_release
        )
        listener.start()

        self.start_visual_loop()

        while self.alive:
            # self.inputs[self.input_neurons] += self.live_input_vector
            self.step(tick)
            tick += 1

            if tick % granularity == 0:
                np.copyto(
                    self.viz_buffer,
                    self.membrane_potentials.reshape(self.shape)
                )

                try:
                    self.viz_q.put_nowait((self.viz_buffer.copy(), tick))
                except queue.Full:
                    # Drop oldest then enqueue latest to reduce latency
                    try:
                        self.viz_q.get_nowait()
                    except queue.Empty:
                        pass
                    finally:
                        try:
                            self.viz_q.put_nowait((self.membrane_potentials.reshape(self.shape), tick))
                        except queue.Full:
                            pass
            time.sleep(speed)
        listener.stop()

    def step(self, tick):
        self.membrane_potentials += self.inputs

        self.inputs.fill(0)

        self.membrane_potentials[(tick - self.last_spiked) == self.SPIKE_PERIOD] = self.RESTING_MP 

        neurons_to_spike = np.where(self.membrane_potentials > self.SPIKE_THRESHOLD)[0]
        self.spike(tick, neurons_to_spike)

        neurons_to_decay = np.where(self.membrane_potentials <= self.SPIKE_THRESHOLD)[0]
        self.decay(neurons_to_decay)

    def spike(self, tick, neurons):
        last_spikes = self.last_spiked[neurons]
        expired = (tick - last_spikes) > self.SPIKE_PERIOD
        expired_neurons = neurons[expired]
        self.last_spiked[expired_neurons] = tick

        self.stdp(tick, neurons)

        children = self.weights.get_neighbors(neurons)
        self.inputs[children.ravel()] += self.weights[
            np.broadcast_to(neurons[:, None], children.shape).ravel(), 
            children.ravel()
        ]

    def decay(self, neurons):
        self.membrane_potentials[neurons] += (self.RESTING_MP - self.membrane_potentials[neurons]) * self.DECAY_RATE

    def stdp(self, tick, neurons):
        children = self.weights.get_neighbors(neurons)
        do_hebb = ~((self.last_spiked[children] == 0) | (self.last_spiked[children] == tick))
        hebb_neurons = np.broadcast_to(neurons[:, None], children.shape)[do_hebb]
        children = children[do_hebb]
        if np.any(children):
            tick_delta = np.abs(tick - self.last_spiked[children])
            decay_deltas = -self.LEARNING_RATE * tick_delta**-3
            self.weights.at[hebb_neurons, children.ravel()] <<= decay_deltas

    def rstdp(self, tick):
        pass


import cupy as cp
from weights_cuda import WeightMatrixCUDA
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
        self.membrane_potentials = cp.empty((self.neuron_count, ), dtype=np.float32)
        self.membrane_potentials.fill(self.RESTING_MP)

        self.mp_logs = cp.zeros((self.neuron_count, 0), dtype=np.float32)
        self.last_spiked = cp.zeros((self.neuron_count, ))

        self.alive = True

    def _setup_lifetime(self, lifetime: int):
        self.lifetime = lifetime
        if lifetime < 0:
            return

        self.mp_logs = np.zeros((self.neuron_count, self.lifetime), dtype=np.float32)

    def set_input_neurons(self, input_list: list): 
        if input_list == None:
            return
        self.input_neurons = np.array(input_list)

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













