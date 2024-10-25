import numpy as np

class AdaptiveQuantumInspiredHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hm_size = 20
        self.hmcr = 0.85
        self.par = 0.35
        self.bw = 0.05
        self.mutation_prob = 0.1
        self.elite_fraction = 0.2
        self.theta_min = -np.pi / 4
        self.theta_max = np.pi / 4
        self.memory_adaptive = True  # Enhanced memory consideration
        self.diversity_adjustment = True  # Diversity-driven mutation adjustment

    def initialize_harmony_memory(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))

    def evaluate_harmonies(self, harmonies, func):
        return np.array([func(harmony) for harmony in harmonies])

    def update_parameters(self, iteration, max_iterations):
        self.hmcr = 0.85 - 0.35 * (iteration / max_iterations)
        self.par = 0.35 + 0.15 * (iteration / max_iterations)
        self.bw = 0.05 * (1 - iteration / max_iterations)
        self.theta = self.theta_min + (self.theta_max - self.theta_min) * (iteration / max_iterations)
        if self.memory_adaptive:
            self.hwcr = np.clip(self.hmcr + 0.05 * np.sin(iteration / max_iterations * 2 * np.pi), 0.75, 0.95)
        if self.diversity_adjustment:
            diversity = np.std(self.harmony_memory, axis=0).mean()
            self.par += 0.05 * (0.3 - diversity)

    def quantum_rotation(self, harmony):
        new_harmony = np.copy(harmony)
        for i in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                rotation_angle = np.random.uniform(self.theta_min, self.theta_max)
                new_harmony[i] += rotation_angle
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony

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

                new_harmony = self.quantum_rotation(new_harmony)
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