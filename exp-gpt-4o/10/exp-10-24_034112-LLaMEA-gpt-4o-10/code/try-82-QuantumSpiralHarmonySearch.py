import numpy as np

class QuantumSpiralHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hm_size = 25
        self.hmcr = 0.9
        self.par = 0.35  # Slightly higher pitch adjustment ratio
        self.bw = 0.03  # Adjusted bandwidth for improved local tuning
        self.mutation_prob = 0.15
        self.elite_fraction = 0.2  # Adjusted elite fraction for diversity
        self.theta_min = -np.pi / 4
        self.theta_max = np.pi / 4
        self.adaptive_diversity_control = True
        self.momentum_factor = 0.8  # Adjusted for better exploration
        self.local_search_prob = 0.1  # Increased local search probability
        self.spiral_factor = 0.5  # Spiral dynamics factor

    def initialize_harmony_memory(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))

    def evaluate_harmonies(self, harmonies, func):
        return np.array([func(harmony) for harmony in harmonies])

    def update_parameters(self, iteration, max_iterations):
        chaos = np.sin(self.spiral_factor * np.pi * iteration / max_iterations) ** 2
        self.hmcr = 0.9 - 0.05 * chaos
        self.par = 0.35 + 0.05 * chaos
        self.bw = 0.03 * (1 - chaos)
        self.theta = self.theta_min + (self.theta_max - self.theta_min) * chaos
        if self.adaptive_diversity_control:
            diversity = np.std(self.harmony_memory, axis=0).mean()
            self.par += 0.04 * (0.1 - diversity)

    def quantum_rotation(self, harmony):
        new_harmony = np.copy(harmony)
        for i in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                rotation_angle = np.random.uniform(self.theta_min, self.theta_max)
                new_harmony[i] += rotation_angle
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony

    def spiral_dynamics(self, harmony, func):
        r = 0.1  # Spiral dynamic radius
        for i in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                harmony[i] += r * np.sin(self.spiral_factor * 2 * np.pi * np.random.rand())
                harmony[i] = np.clip(harmony[i], self.lower_bound, self.upper_bound)
        return harmony if func(harmony) < func(harmony) else harmony

    def __call__(self, func):
        self.harmony_memory = self.initialize_harmony_memory()
        harmony_values = self.evaluate_harmonies(self.harmony_memory, func)
        evaluations = self.hm_size
        max_iterations = self.budget // self.hm_size
        num_elites = int(self.elite_fraction * self.hm_size)

        for iteration in range(max_iterations):
            self.update_parameters(iteration, max_iterations)
            elite_indices = np.argsort(harmony_values)[:num_elites]
            elite_harmonies = self.harmony_memory[elite_indices]

            for _ in range(self.hm_size):
                new_harmony = np.copy(elite_harmonies[np.random.randint(num_elites)])
                for i in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        new_harmony[i] = self.harmony_memory[np.random.randint(self.hm_size)][i]
                        if np.random.rand() < self.par:
                            new_harmony[i] += self.bw * (np.random.rand() - 0.5) * 2
                            new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
                    else:
                        new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)

                if np.random.rand() < self.momentum_factor:
                    new_harmony = self.quantum_rotation(new_harmony)

                if np.random.rand() < self.local_search_prob:
                    new_harmony = self.spiral_dynamics(new_harmony, func)

                new_value = func(new_harmony)
                evaluations += 1

                if new_value < np.max(harmony_values):
                    worst_index = np.argmax(harmony_values)
                    self.harmony_memory[worst_index] = new_harmony
                    harmony_values[worst_index] = new_value

                if evaluations >= self.budget:
                    break

        best_index = np.argmin(harmony_values)
        return self.harmony_memory[best_index]