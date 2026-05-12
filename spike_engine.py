import matplotlib.animation as animation
import plotly.graph_objects as go, time
from IPython.display import display
from dataclasses import dataclass
from pathlib import Path
from weights import WeightMatrix
import matplotlib.pyplot as plt
from pynput import keyboard
import threading, queue
import ipywidgets as w
from tqdm import tqdm
from PIL import Image
import bz2
import gzip
import lzma
import numpy as np
import warnings
import io


DEFAULT_MAX_LOG_BYTES = 512 * 1024 * 1024


def _format_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


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

    def __init__(
        self, 
        network: dict, 
        shape: tuple,
        rank: int = None, 
        weight_initializer: callable = np.random.normal, 
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

    def _setup_lifetime(
        self,
        lifetime: int,
        *,
        allocate_logs: bool = False,
        max_log_bytes: int | None = DEFAULT_MAX_LOG_BYTES,
    ):
        self.lifetime = int(lifetime)
        if self.lifetime < 0 or not allocate_logs:
            self.mp_logs = np.zeros((self.neuron_count, 0), dtype=np.float32)
            return

        required_bytes = self.neuron_count * self.lifetime * np.dtype(np.float32).itemsize
        if max_log_bytes is not None and required_bytes > max_log_bytes:
            raise MemoryError(
                "Refusing to allocate membrane log "
                f"({self.neuron_count} neurons x {self.lifetime} ticks, "
                f"{_format_bytes(required_bytes)}). Static recording streams "
                "to disk; only allocate mp_logs when explicitly needed with a "
                "larger max_log_bytes budget."
            )

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
        if input_list is None:
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

    def start_static_record(
        self,
        input_spikes: np.ndarray,
        lifetime: int,
        filename: str,
        record_membrane: bool = True,
        record_stride: int = 1,
        compression: str | None = "auto",
        compression_level: int | None = None,
    ):
        if lifetime < 0:
            raise ValueError("lifetime must be >= 0 for static recording.")
        if record_stride < 1:
            raise ValueError("record_stride must be >= 1.")
        tick = 0
        input_neurons = getattr(self, "input_neurons", None)
        if input_neurons is None or len(input_neurons) == 0:
            print("Set input neurons before starting the simulation.")
            return
        self._setup_lifetime(lifetime, allocate_logs=False)
        self.recording = True
        comp, out_path = self._resolve_record_compression(filename, compression)
        try:
            with tqdm(total=self.lifetime) as progress:
                with self._open_record_file(out_path, comp, compression_level) as f:
                    f.write(self.neuron_count.to_bytes(4, "big"))
                    while tick < self.lifetime:
                        self.inputs[self.input_neurons] += input_spikes[tick]
                        self.step(tick)
                        if record_membrane and (tick % record_stride == 0):
                            f.write(self.membrane_potentials.tobytes())
                        tick += 1
                        progress.update(1)
            print(f"Recording saved: {out_path}")
        finally:
            self.recording = False

    def _resolve_record_compression(self, filename: str, compression: str | None) -> tuple[str | None, str]:
        path = str(filename)
        if compression is None or compression is False or str(compression).lower() == "none":
            return None, path

        comp = str(compression).lower()
        ext_map = {
            ".xz": "xz",
            ".lzma": "xz",
            ".gz": "gz",
            ".gzip": "gz",
            ".bz2": "bz2",
        }
        if comp == "auto":
            suffix = Path(path).suffix.lower()
            return ext_map.get(suffix), path

        if comp in ("xz", "lzma"):
            comp = "xz"
        elif comp in ("gz", "gzip"):
            comp = "gz"
        elif comp in ("bz2", "bzip2"):
            comp = "bz2"
        else:
            raise ValueError(f"Unsupported compression: {compression}")

        ext = {"xz": ".xz", "gz": ".gz", "bz2": ".bz2"}[comp]
        if not path.lower().endswith(ext):
            path = path + ext
        return comp, path

    def _open_record_file(self, path: str, compression: str | None, level: int | None):
        if compression is None:
            return open(path, "wb")
        if compression == "xz":
            preset = level if level is not None else 6
            return lzma.open(path, "wb", preset=preset)
        if compression == "gz":
            compresslevel = level if level is not None else 6
            return gzip.open(path, "wb", compresslevel=compresslevel)
        if compression == "bz2":
            compresslevel = level if level is not None else 6
            return bz2.open(path, "wb", compresslevel=compresslevel)
        raise ValueError(f"Unsupported compression: {compression}")

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
