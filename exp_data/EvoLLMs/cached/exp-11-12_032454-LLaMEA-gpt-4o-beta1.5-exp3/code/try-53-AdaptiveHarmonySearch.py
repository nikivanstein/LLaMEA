import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.harmony_memory_size = max(5, int(budget / (15 * dim)))  # heuristic for harmony memory size
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par_min = 0.1  # Minimum Pitch Adjustment Rate
        self.par_max = 0.99  # Maximum Pitch Adjustment Rate
        self.bw = 0.02 * (self.ub - self.lb)  # Bandwidth for pitch adjustment

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(self.lb, self.ub, (self.harmony_memory_size, self.dim))
        fitness = np.array([func(harmony) for harmony in harmony_memory])
        num_evaluations = self.harmony_memory_size

        best_idx = np.argmin(fitness)
        best_harmony = harmony_memory[best_idx]
        best_fitness = fitness[best_idx]

        while num_evaluations < self.budget:
            for _ in range(self.harmony_memory_size):
                if num_evaluations >= self.budget:
                    break

                # Generate new harmony
                new_harmony = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
                for j in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        # Consider harmony memory
                        new_harmony[j] = harmony_memory[np.random.randint(self.harmony_memory_size), j]
                        if np.random.rand() < self.current_par(num_evaluations):
                            # Pitch adjustment
                            new_harmony[j] += self.bw * (2 * np.random.rand() - 1)
                            new_harmony[j] = np.clip(new_harmony[j], self.lb, self.ub)
                    else:
                        # Random selection
                        new_harmony[j] = np.random.uniform(self.lb, self.ub)

                # Evaluate new harmony
                new_fitness = func(new_harmony)
                num_evaluations += 1

                # Update harmony memory
                if new_fitness < max(fitness):
                    worst_idx = np.argmax(fitness)
                    harmony_memory[worst_idx] = new_harmony
                    fitness[worst_idx] = new_fitness

                    if new_fitness < best_fitness:
                        best_harmony = new_harmony
                        best_fitness = new_fitness

        return best_harmony, best_fitness

    def current_par(self, num_evaluations):
        # Adaptive Pitch Adjustment Rate
        return self.par_min + (self.par_max - self.par_min) * (num_evaluations / self.budget)