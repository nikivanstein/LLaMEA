import numpy as np

class MultiSwarmHybridHarmonyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_harmony_memory(size):
            return np.random.uniform(-5.0, 5.0, (size, self.dim))

        def generate_new_harmony(harmony_memory):
            new_harmony = np.random.uniform(-5.0, 5.0, self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    new_harmony[i] = np.random.choice(harmony_memory[:, i])
            return new_harmony

        def local_search(harmony):
            step_size = 0.2 + 0.1 * np.cos(10*np.pi*np.arange(self.dim)/self.dim)  # Dynamic step size adjustment
            for i in range(self.dim):
                perturbed_harmony = np.copy(harmony)
                perturbed_harmony[i] += step_size[i]
                if func(perturbed_harmony) < func(harmony):
                    harmony[i] += step_size[i]
                elif func(perturbed_harmony) > func(harmony):
                    harmony[i] -= step_size[i]
            return harmony

        harmony_memory = initialize_harmony_memory(10)
        swarm_position = np.random.uniform(-5.0, 5.0, (self.dim,))
        swarm_velocity = np.random.uniform(-1.0, 1.0, (self.dim,))

        for _ in range(int(self.budget * 0.6)):  # Harmony Search
            new_harmony = generate_new_harmony(harmony_memory)
            new_harmony = local_search(new_harmony)
            if func(new_harmony) < np.min(func(harmony_memory)):
                idx = np.argmin(func(harmony_memory))
                harmony_memory[idx] = new_harmony

        for _ in range(int(self.budget * 0.4)):  # Multi-Swarm PSO
            inertia_weight = 0.9
            cognitive_weight = 1.5
            social_weight = 1.5
            swarm_velocity = inertia_weight * swarm_velocity + cognitive_weight * np.random.rand() * (harmony_memory[np.argmin(func(harmony_memory))] - swarm_position) + social_weight * np.random.rand() * (swarm_position - swarm_position)
            swarm_position = swarm_position + swarm_velocity
            if func(swarm_position) < np.min(func(harmony_memory)):
                harmony_memory[np.argmin(func(harmony_memory))] = swarm_position

        best_solution = harmony_memory[np.argmin(func(harmony_memory))]
        return best_solution