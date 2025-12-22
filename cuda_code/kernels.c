#define NEIGHB_COUNT <<NEIGHB_COUNT_SUB>>
#define k <<K_SUB>>

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

extern "C" __global__
void step_kernel(
    int tick,
    const int SPIKE_PERIOD,
    const float SPIKE_THRESHOLD,
    const float LEARNING_RATE,
    const float DECAY_RATE,
    const float RESTING_MP,
    float* __restrict__ U,
    float* __restrict__ V,
    const int* __restrict__ neighbors,
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
        const int neighbor_base = neuron_thread_id * NEIGHB_COUNT;
        for (int c = 0; c < NEIGHB_COUNT; ++c) {
            int child = neighbors[neighbor_base + c];
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


















