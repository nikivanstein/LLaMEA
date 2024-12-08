import numpy as np

class DynamicMultiPopulationQuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hm_size = 30  # Slight increase in harmony memory size
        self.hmcr = 0.85  # Adjusted harmony memory consideration rate
        self.par = 0.25  # Slightly decreased pitch adjustment ratio
        self.bw = 0.015  # Narrower bandwidth for local search precision
        self.mutation_prob = 0.1  # Reduced mutation probability
        self.elite_fraction = 0.2  # More focus on non-elite harmonies
        self.theta_min = -np.pi / 3  # Increased rotation angle range
        self.theta_max = np.pi / 3
        self.adaptive_diversity_control = True
        self.momentum_factor = 0.8  # Reduced momentum factor
        self.local_search_prob = 0.1  # Increased chance of local search
        self.chaos_factor = 0.9  # Higher chaos factor for dynamic tuning
        self.num_populations = 2  # Multiple populations for diversity

    def initialize_harmony_memory(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.num_populations, self.hm_size, self.dim))

    def evaluate_harmonies(self, harmonies, func):
        return np.array([[func(harmony) for harmony in population] for population in harmonies])

    def update_parameters(self, iteration, max_iterations):
        chaos = np.sin(self.chaos_factor * np.pi * iteration / max_iterations) ** 2
        self.hmcr = 0.85 - 0.1 * chaos
        self.par = 0.25 + 0.1 * chaos
        self.bw = 0.015 * (1 - chaos)
        self.theta = self.theta_min + (self.theta_max - self.theta_min) * chaos
        if self.adaptive_diversity_control:
            diversity = np.std(self.harmony_memory, axis=(1, 2)).mean()
            self.par += 0.05 * (0.1 - diversity)
        self.momentum_factor = 0.8 - 0.1 * chaos

    def quantum_rotation(self, harmony):
        new_harmony = np.copy(harmony)
        for i in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                rotation_angle = np.random.uniform(self.theta_min, self.theta_max)
                new_harmony[i] += rotation_angle
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony

    def local_search(self, harmony, func):
        perturbation = np.random.normal(0, 0.1, size=self.dim)
        new_harmony = harmony + perturbation
        new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
        if func(new_harmony) < func(harmony):
            return new_harmony
        return harmony

    def __call__(self, func):
        self.harmony_memory = self.initialize_harmony_memory()
        harmony_values = self.evaluate_harmonies(self.harmony_memory, func)
        evaluations = self.hm_size * self.num_populations
        max_iterations = self.budget // (self.hm_size * self.num_populations)
        num_elites = int(self.elite_fraction * self.hm_size)

        for iteration in range(max_iterations):
            self.update_parameters(iteration, max_iterations)
            for population in range(self.num_populations):
                elite_indices = np.argsort(harmony_values[population])[:num_elites]
                elite_harmonies = self.harmony_memory[population][elite_indices]

                for _ in range(self.hm_size):
                    new_harmony = np.copy(elite_harmonies[np.random.randint(num_elites)])
                    for i in range(self.dim):
                        if np.random.rand() < self.hmcr:
                            new_harmony[i] = self.harmony_memory[population][np.random.randint(self.hm_size)][i]
                            if np.random.rand() < self.par:
                                new_harmony[i] += self.bw * (np.random.rand() - 0.5) * 2
                                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
                        else:
                            new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)

                    if np.random.rand() < self.momentum_factor:
                        new_harmony = self.quantum_rotation(new_harmony)

                    if np.random.rand() < self.local_search_prob:
                        new_harmony = self.local_search(new_harmony, func)

                    new_value = func(new_harmony)
                    evaluations += 1

                    if new_value < np.max(harmony_values[population]):
                        worst_index = np.argmax(harmony_values[population])
                        self.harmony_memory[population][worst_index] = new_harmony
                        harmony_values[population][worst_index] = new_value

                    if evaluations >= self.budget:
                        break

        best_value = np.inf
        best_solution = None
        for population in range(self.num_populations):
            best_index = np.argmin(harmony_values[population])
            if harmony_values[population][best_index] < best_value:
                best_value = harmony_values[population][best_index]
                best_solution = self.harmony_memory[population][best_index]

        return best_solution