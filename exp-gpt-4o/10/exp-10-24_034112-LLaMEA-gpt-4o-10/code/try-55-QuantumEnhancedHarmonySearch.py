import numpy as np

class QuantumEnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hm_size = 25
        self.hmcr = 0.95  # Increased harmony memory considering rate for better use of knowledge
        self.par = 0.35  # Slightly increased pitch adjustment ratio
        self.bw = 0.015  # Decreased bandwidth for precise local search
        self.mutation_prob = 0.1
        self.elite_fraction = 0.2
        self.theta_min = -np.pi / 3
        self.theta_max = np.pi / 3
        self.adaptive_diversity_control = True
        self.momentum_factor = 0.9  # Increased momentum for faster convergence
        self.local_search_prob = 0.07  # Increased probability for local search
        self.chaos_factor = 0.65  # Slight adjustment in chaos factor
        self.sde_intensity = 0.1  # New Stochastic Differential Equation component

    def initialize_harmony_memory(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))

    def evaluate_harmonies(self, harmonies, func):
        return np.array([func(harmony) for harmony in harmonies])

    def update_parameters(self, iteration, max_iterations):
        chaos = np.sin(self.chaos_factor * np.pi * (iteration + 1) / max_iterations) ** 2
        self.hmcr = 0.95 - 0.05 * chaos
        self.par = 0.35 + 0.1 * chaos
        self.bw = 0.015 * (1 - chaos)
        self.theta = self.theta_min + (self.theta_max - self.theta_min) * chaos
        if self.adaptive_diversity_control:
            diversity = np.std(self.harmony_memory, axis=0).mean()
            self.par += 0.05 * (0.1 - diversity)
        self.momentum_factor = 0.9 - 0.05 * chaos

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

    def stochastic_differential_perturbation(self, harmony):
        noise = np.random.normal(0, self.sde_intensity, size=self.dim)
        return np.clip(harmony + noise, self.lower_bound, self.upper_bound)

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
                    new_harmony = self.local_search(new_harmony, func)

                # Apply stochastic differential perturbation
                new_harmony = self.stochastic_differential_perturbation(new_harmony)

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