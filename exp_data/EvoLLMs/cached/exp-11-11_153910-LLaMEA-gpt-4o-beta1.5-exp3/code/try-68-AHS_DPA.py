import numpy as np

class AHS_DPA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hm_size = 30  # Harmony memory size
        self.hmcr = 0.9    # Harmony memory consideration rate
        self.par = 0.3     # Pitch adjustment rate
        self.bw = 0.02     # Bandwidth for pitch adjustment
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))
        self.fitness = np.full(self.hm_size, float('inf'))
        self.best_harmony = None
        self.best_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def dynamic_parameter_adjustment(self):
        if self.evaluations > self.budget * 0.7:
            self.hmcr = 0.95
            self.par = 0.4
            self.bw = 0.01

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.hm_size):
            self.fitness[i] = self.evaluate(func, self.harmony_memory[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_harmony = self.harmony_memory[i]

        while self.evaluations < self.budget:
            self.dynamic_parameter_adjustment()
            for _ in range(self.hm_size):
                new_harmony = np.copy(self.harmony_memory[np.random.randint(self.hm_size)])
                for i in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        if np.random.rand() < self.par:
                            new_harmony[i] += self.bw * (2 * np.random.rand() - 1)
                        new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
                    else:
                        new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                
                new_fitness = self.evaluate(func, new_harmony)
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_harmony = new_harmony
                
                # Replace worst harmony if new is better
                worst_idx = np.argmax(self.fitness)
                if new_fitness < self.fitness[worst_idx]:
                    self.harmony_memory[worst_idx] = new_harmony
                    self.fitness[worst_idx] = new_fitness

        return self.best_harmony