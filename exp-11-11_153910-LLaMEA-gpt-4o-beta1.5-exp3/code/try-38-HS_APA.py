import numpy as np

class HS_APA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, 
                                                (self.harmony_memory_size, self.dim))
        self.fitness = np.full(self.harmony_memory_size, float('inf'))
        self.best_harmony = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.harmony_memory_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.1
        self.bandwidth = 0.1

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def adapt_pitch(self, harmony):
        if np.random.rand() < self.pitch_adjustment_rate:
            adjustment = self.bandwidth * (2 * np.random.rand(self.dim) - 1)
            harmony += adjustment
        return np.clip(harmony, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.harmony_memory_size):
            self.fitness[i] = self.evaluate(func, self.harmony_memory[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_harmony = self.harmony_memory[i]

        while self.evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[j] = self.harmony_memory[idx, j]
                else:
                    new_harmony[j] = np.random.uniform(self.lower_bound, self.upper_bound)

            new_harmony = self.adapt_pitch(new_harmony)
            new_harmony_fitness = self.evaluate(func, new_harmony)
            
            if new_harmony_fitness < np.max(self.fitness):
                worst_idx = np.argmax(self.fitness)
                self.harmony_memory[worst_idx] = new_harmony
                self.fitness[worst_idx] = new_harmony_fitness
                if new_harmony_fitness < self.best_fitness:
                    self.best_fitness = new_harmony_fitness
                    self.best_harmony = new_harmony

        return self.best_harmony