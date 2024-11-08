import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hms = 20
        self.hmcr = 0.85
        self.par = 0.35
        self.bandwidth = 0.1
        self.harmony_memory = None
        self.fitness_memory = None
        self.evaluations = 0

    def initialize_harmony_memory(self):
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim))
        self.fitness_memory = np.full(self.hms, float('inf'))

    def evaluate_fitness(self, func, harmony):
        if self.evaluations < self.budget:
            fitness = func(harmony)
            self.evaluations += 1
            return fitness
        return float('inf')

    def improvise_new_harmony(self):
        new_harmony = np.empty(self.dim)
        hmcr_choices = np.random.rand(self.dim) < self.hmcr
        par_choices = np.random.rand(self.dim) < self.par
        random_indices = np.random.randint(self.hms, size=self.dim)
        random_adjustments = self.bandwidth * np.random.uniform(-1, 1, size=self.dim)

        for i in range(self.dim):
            if hmcr_choices[i]:
                new_harmony[i] = self.harmony_memory[random_indices[i], i]
                if par_choices[i]:
                    new_harmony[i] += random_adjustments[i]
            else:
                new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony

    def update_harmony_memory(self, new_harmony, new_fitness):
        worst_index = np.argmax(self.fitness_memory)
        if new_fitness < self.fitness_memory[worst_index]:
            self.harmony_memory[worst_index] = new_harmony
            self.fitness_memory[worst_index] = new_fitness

    def __call__(self, func):
        self.initialize_harmony_memory()
        self.fitness_memory = np.array([self.evaluate_fitness(func, harmony) for harmony in self.harmony_memory])

        while self.evaluations < self.budget:
            new_harmony = self.improvise_new_harmony()
            new_fitness = self.evaluate_fitness(func, new_harmony)
            self.update_harmony_memory(new_harmony, new_fitness)

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index], self.fitness_memory[best_index]