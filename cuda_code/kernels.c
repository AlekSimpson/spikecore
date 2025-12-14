#define NEIGHB_COUNT <<NEIGHB_COUNT_SUB>>
#define k <<K_SUB>>

typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;


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

__device__
uint64_t splitmix64(uint64_t x, uint64_t mask) {
    uint64_t A = 0x9E3779B97F4A7C15;
    uint64_t B = 0xBF58476D1CE4E5B9;
    uint64_t C = 0x94D049BB133111EB;

    uint64_t z = (x + A) & mask;
    z = z ^ (z >> ((uint64_t)30));
    z = (z * B) & mask;
    z = z ^ (z >> ((uint64_t)27));
    z = (z * C) & mask;
    z = z ^ (z >> (uint64_t)31);
    return z ^ mask;
}

__device__
void get_neighbors(
    const int64_t* __restrict__ bf_table,
    uint64_t bf_salt,
    uint64_t bf_MASK64,
    const uint64_t bf_key_amount,
    int64_t* results,
    int64_t neuron
) {
    int64_t key;
    int64_t h1, h2, h3;
    for (int64_t kk = 0; kk < NEIGHB_COUNT; ++kk) {
        key = ((neuron + kk)*(neuron + kk + 1)) / 2 + kk;
	key = (key ^ bf_salt) & bf_MASK64;
	h1 = splitmix64(key, bf_MASK64) % bf_key_amount;
	h2 = splitmix64(key + ((uint64_t)1), bf_MASK64) % bf_key_amount;
	h3 = splitmix64(key + ((uint64_t)0x9D), bf_MASK64) % bf_key_amount;

	results[kk] = bf_table[h1] ^ bf_table[h2] ^ bf_table[h3];
    }
}

extern "C" __global__
void step_kernel(
    int tick,
    const int SPIKE_PERIOD,
    const float SPIKE_THRESHOLD,
    const float LEARNING_RATE,
    const float DECAY_RATE,
    const float RESTING_MP,
    uint64_t bf_salt,
    uint64_t bf_MASK64,
    const int bf_key_amount,
    float* __restrict__ U,
    float* __restrict__ V,
    const int64_t* __restrict__ bf_table,
    int neuron_count,
    float* __restrict__ inputs,
    float* __restrict__ membrane_potentials,
    int* __restrict__ last_spiked
) {
    int neuron_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (neuron_thread_id >= neuron_count) return; // one thread per neuron

    // todo: would updating the input neuron updates in the kernel be faster?

    membrane_potentials[neuron_thread_id] = membrane_potentials[neuron_thread_id] + inputs[neuron_thread_id];
    inputs[neuron_thread_id] = 0;

    int time_last_spiked = last_spiked[neuron_thread_id];
    if ((tick - time_last_spiked) == SPIKE_PERIOD) {
        membrane_potentials[neuron_thread_id] = RESTING_MP;
        return;
    }

    if (membrane_potentials[neuron_thread_id] > SPIKE_THRESHOLD) {
        // spike
        if ((tick - time_last_spiked) > SPIKE_PERIOD) {
            last_spiked[neuron_thread_id] = tick;
        }

        // stdp hebb rule
	int64_t children[NEIGHB_COUNT];
        get_neighbors(bf_table, bf_salt, bf_MASK64, (uint64_t)bf_key_amount, children, neuron_thread_id);

        for (int c = 0; c < NEIGHB_COUNT; ++c) {
            int64_t child = children[c];
	    // printf("child %d is: %d\n", c, child);

            if (!(last_spiked[child] == 0 || last_spiked[child] == tick)) {
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

            const float* u = U + (size_t)neuron_thread_id * k;
	    const float* v = V + (size_t)child * k;
	    float dot = 0.0f;
	    for (int i = 0; i < k; ++i) {
		dot += u[i] * v[i];
	    }
	    atomicAdd(&inputs[child], dot);
        }
        return;
    }

    // otherwise decay and end
    float neuron_mp = membrane_potentials[neuron_thread_id];
    membrane_potentials[neuron_thread_id] = neuron_mp + (RESTING_MP - neuron_mp) * DECAY_RATE;
}



















