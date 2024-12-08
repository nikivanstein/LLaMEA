import numpy as np

class AQHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.harmony_memory_size = 10 * dim
        self.harmony_memory = []
        self.scale_factor = 0.7
        self.crossover_prob = 0.85
        self.quantum_prob = 0.5

    def __call__(self, func):
        # Initialize harmony memory
        for _ in range(self.harmony_memory_size):
            harmony = self.lower_bound + np.random.rand(self.dim) * (self.upper_bound - self.lower_bound)
            fitness = func(harmony)
            self.evaluations += 1
            self.harmony_memory.append((harmony, fitness))

        # Sort harmony memory by fitness
        self.harmony_memory.sort(key=lambda x: x[1])
        best_harmony, best_fitness = self.harmony_memory[0]

        while self.evaluations < self.budget:
            new_harmony = np.empty(self.dim)

            for i in range(self.dim):
                if np.random.rand() < self.quantum_prob:
                    # Quantum-inspired random selection
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx][0][i]
                else:
                    # Generate a trial harmony based on the best harmony
                    if np.random.rand() < self.crossover_prob:
                        new_harmony[i] = best_harmony[i] + self.scale_factor * (np.random.rand() - 0.5)
                    else:
                        new_harmony[i] = self.lower_bound + np.random.rand() * (self.upper_bound - self.lower_bound)

            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_fitness = func(new_harmony)
            self.evaluations += 1

            # Update harmony memory if new harmony is better
            if new_fitness < self.harmony_memory[-1][1]:
                self.harmony_memory[-1] = (new_harmony, new_fitness)
                self.harmony_memory.sort(key=lambda x: x[1])
                if new_fitness < best_fitness:
                    best_harmony = new_harmony
                    best_fitness = new_fitness

        return best_harmony, best_fitness