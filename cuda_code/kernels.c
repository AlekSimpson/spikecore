#define NEIGHB_COUNT <<NEIGHB_COUNT_SUB>>
#define k <<K_SUB>>

typedef long long int64_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

__device__ __forceinline__
float apply_decay(float mp, float resting, float decay_rate, int dt) {
    if (dt <= 0) {
        return mp;
    }
    if (dt == 1) {
        return resting + (mp - resting) * (1.0f - decay_rate);
    }
    float decay = powf(1.0f - decay_rate, (float)dt);
    return resting + (mp - resting) * decay;
}


__device__
void update_weight_matrix(
    float* __restrict__ U,
    float* __restrict__ V,
    int i,
    int j,
    float delta,
    float lr,
    float l2_reg

) {
    float u_anchor[k];
    float v_anchor[k];
    
    for (int d = 0; d < k; d++) {
        u_anchor[d] = U[i * k + d];
        v_anchor[d] = V[j * k + d];
    }
    
    float den_v = l2_reg;
    float den_u = l2_reg;
    
    for (int d = 0; d < k; d++) {
        den_v += v_anchor[d] * v_anchor[d];
        den_u += u_anchor[d] * u_anchor[d];
    }

    for (int d = 0; d < k; d++) {
        float du = lr * (delta * (v_anchor[d] / den_v) - l2_reg * (u_anchor[d] - u_anchor[d]));
        float dv = lr * (delta * (u_anchor[d] / den_u) - l2_reg * (v_anchor[d] - v_anchor[d]));
        
	atomicAdd(&U[i * k + d], du);
	atomicAdd(&V[j * k + d], dv);
    }
}

extern "C" __global__
void add_active_kernel(
    const int* __restrict__ indices,
    int n_indices,
    int tick,
    int* __restrict__ active,
    int* __restrict__ active_count,
    int* __restrict__ active_gen
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_indices) return;

    int neuron = indices[idx];
    int prev = atomicExch(&active_gen[neuron], tick);
    if (prev != tick) {
        int pos = atomicAdd(active_count, 1);
        active[pos] = neuron;
    }
}

extern "C" __global__
void decay_kernel(
    int neuron_count,
    float* __restrict__ membrane_potentials,
    int* __restrict__ last_updated,
    const float RESTING_MP,
    const float DECAY_RATE,
    int tick
) {
    int neuron_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (neuron_thread_id >= neuron_count) return;
    float mp = membrane_potentials[neuron_thread_id];
    int dt = tick - last_updated[neuron_thread_id];
    mp = apply_decay(mp, RESTING_MP, DECAY_RATE, dt);
    membrane_potentials[neuron_thread_id] = mp;
    last_updated[neuron_thread_id] = tick;
}

extern "C" __global__
void step_kernel(
    int tick,
    int next_tick,
    const int SPIKE_PERIOD,
    const float SPIKE_THRESHOLD,
    const float LEARNING_RATE,
    const float DECAY_RATE,
    const float RESTING_MP,
    float* __restrict__ U,
    float* __restrict__ V,
    const int USE_CONSTANT_WEIGHT,
    const float CONSTANT_WEIGHT,
    const int* __restrict__ neighbors,
    int neuron_count,
    float* __restrict__ inputs,
    float* __restrict__ membrane_potentials,
    int* __restrict__ last_spiked,
    int* __restrict__ last_updated,
    const int* __restrict__ active,
    const int* __restrict__ active_count,
    int* __restrict__ next_active,
    int* __restrict__ next_count,
    int* __restrict__ active_gen
) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int count = active_count[0];
    if (thread_id >= count) return;

    int neuron_thread_id = active[thread_id];
    if (neuron_thread_id < 0 || neuron_thread_id >= neuron_count) return;

    // todo: would updating the input neuron updates in the kernel be faster?

    int last_upd = last_updated[neuron_thread_id];
    int dt = tick - last_upd;
    float mp = membrane_potentials[neuron_thread_id];
    mp = apply_decay(mp, RESTING_MP, DECAY_RATE, dt);
    mp = mp + inputs[neuron_thread_id];
    inputs[neuron_thread_id] = 0;

    int time_last_spiked = last_spiked[neuron_thread_id];
    if ((tick - time_last_spiked) == SPIKE_PERIOD) {
        membrane_potentials[neuron_thread_id] = RESTING_MP;
        last_updated[neuron_thread_id] = tick;
        return;
    }

    if (mp > SPIKE_THRESHOLD) {
        // spike
        if ((tick - time_last_spiked) > SPIKE_PERIOD) {
            last_spiked[neuron_thread_id] = tick;
        }

        // stdp hebb rule
        const int neighbor_base = neuron_thread_id * NEIGHB_COUNT;
        for (int c = 0; c < NEIGHB_COUNT; ++c) {
            int child = neighbors[neighbor_base + c];
            // printf("child %d is: %d\n", c, child);

            if (LEARNING_RATE != 0.0f && !(last_spiked[child] == 0 || last_spiked[child] == tick)) {
                float tick_delta = (float)(tick - last_spiked[child]);
                float decay_delta = -LEARNING_RATE * powf(tick_delta, -3);
                update_weight_matrix(
                    U, V,
                    neuron_thread_id,
                    child,
                    decay_delta,
                    0.5,
                    1
                );
            }

            float weight = CONSTANT_WEIGHT;
            if (!USE_CONSTANT_WEIGHT) {
                const float* u = U + (size_t)neuron_thread_id * k;
                const float* v = V + (size_t)child * k;
                float dot = 0.0f;
                for (int i = 0; i < k; ++i) {
                    dot += u[i] * v[i];
                }
                weight = dot;
            }
            atomicAdd(&inputs[child], weight);

            int prev = atomicExch(&active_gen[child], next_tick);
            if (prev != next_tick) {
                int pos = atomicAdd(next_count, 1);
                next_active[pos] = child;
            }
        }
        int prev = atomicExch(&active_gen[neuron_thread_id], next_tick);
        if (prev != next_tick) {
            int pos = atomicAdd(next_count, 1);
            next_active[pos] = neuron_thread_id;
        }
        membrane_potentials[neuron_thread_id] = mp;
        last_updated[neuron_thread_id] = tick;
        return;
    }

    // otherwise decay and end
    membrane_potentials[neuron_thread_id] = mp;
    last_updated[neuron_thread_id] = tick;
}












