#define NEIGHB_COUNT <<NEIGHB_COUNT_SUB>>
#define k <<K_SUB>>


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
        
        U[i * k + d] += du;
        V[j * k + d] += dv;
    }
}

__device__
int splitmix64(int x, unsigned long long mask) {
    int z = (x + ((unsigned int)0x9E3779B97F4A7C15)) & mask;
    z = z ^ (z >> ((unsigned int)30));
    z = (z * ((unsigned int)0xBF58476D1CE4E5B9)) & mask;
    z = z ^ (z >> ((unsigned int)27));
    z = (z * ((unsigned int)0x94D049BB133111EB)) & mask;
    z = z ^ (z >> (unsigned int)31);
    return z ^ mask;
}

__device__
void get_neighbors(
    const int* __restrict__ bf_table,
    unsigned long long bf_salt,
    unsigned long long bf_MASK64,
    const int bf_key_amount,
    int* results,
    int neuron
) {
    int key;
    int h1, h2, h3;
    for (int kk = 0; kk < NEIGHB_COUNT; ++kk) {
        key = ((neuron + kk)*(neuron + kk + 1)) / 2 + kk;
	key = (key ^ bf_salt) & bf_MASK64;
	h1 = splitmix64(key, bf_MASK64) % bf_key_amount;
	h2 = splitmix64(key + 1, bf_MASK64) % bf_key_amount;
	h3 = splitmix64(key + 0x9D, bf_MASK64) % bf_key_amount;

	results[kk] = bf_table[h1] ^ bf_table[h2] ^ bf_table[h3];
    }
}

extern "C" __global__
void step_kernel(
    int tick,
    const int SPIKE_PERIOD,
    const int SPIKE_THRESHOLD,
    const float LEARNING_RATE,
    const float DECAY_RATE,
    const float RESTING_MP,
    unsigned long long bf_salt,
    unsigned long long bf_MASK64,
    const int bf_key_amount,
    float* __restrict__ U,
    float* __restrict__ V,
    const int* __restrict__ bf_table,
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
	int children[NEIGHB_COUNT];
        get_neighbors(bf_table, bf_salt, bf_MASK64, bf_key_amount, children, neuron_thread_id);

        for (int c = 0; c < NEIGHB_COUNT; ++c) {
            int child = children[c];
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
	    inputs[child] += dot;
        }

        return;
    }

    // otherwise decay and end
    float neuron_mp = membrane_potentials[neuron_thread_id];
    membrane_potentials[neuron_thread_id] = neuron_mp + (RESTING_MP - neuron_mp) * DECAY_RATE;
}



















