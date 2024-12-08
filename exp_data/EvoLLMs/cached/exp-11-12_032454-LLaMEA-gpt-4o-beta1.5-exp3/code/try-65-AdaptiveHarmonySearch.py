import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.harmony_memory_size = max(5, int(budget / (10 * dim)))  # Memory size heuristic
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par_min = 0.2  # Minimum Pitch Adjustment Rate
        self.par_max = 0.9  # Maximum Pitch Adjustment Rate
        self.bw = 0.02  # Bandwidth for pitch adjustment

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(self.lb, self.ub, (self.harmony_memory_size, self.dim))
        fitness = np.array([func(ind) for ind in harmony_memory])
        num_evaluations = self.harmony_memory_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]

        while num_evaluations < self.budget:
            for _ in range(self.harmony_memory_size):
                if num_evaluations >= self.budget:
                    break

                # Generate new harmony
                new_harmony = np.zeros(self.dim)
                for d in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        # Memory consideration
                        idx = np.random.randint(self.harmony_memory_size)
                        new_harmony[d] = harmony_memory[idx, d]
                        if np.random.rand() < self._adaptive_par(fitness, best_fitness):
                            # Pitch adjustment
                            new_harmony[d] += self.bw * (np.random.rand() - 0.5)
                    else:
                        # Random selection
                        new_harmony[d] = np.random.uniform(self.lb, self.ub)

                # Clipping
                new_harmony = np.clip(new_harmony, self.lb, self.ub)

                # Evaluate new harmony
                new_fitness = func(new_harmony)
                num_evaluations += 1

                # Update harmony memory if new harmony is better
                if new_fitness < np.max(fitness):
                    worst_idx = np.argmax(fitness)
                    harmony_memory[worst_idx] = new_harmony
                    fitness[worst_idx] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness

        return harmony_memory[np.argmin(fitness)], best_fitness

    def _adaptive_par(self, fitness, best_fitness):
        # Adaptive pitch adjustment rate based on memory diversity
        diversity = np.std(fitness)
        norm_diversity = (diversity - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-10)
        return self.par_min + (self.par_max - self.par_min) * norm_diversity