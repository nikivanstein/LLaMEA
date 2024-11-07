import numpy as np

class ImprovedHarmonySearch:
    def __init__(self, budget, dim, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])

        for _ in range(self.budget - len(population)):
            if np.random.uniform() < self.mutation_rate:
                mutation_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                new_harmony = population[np.random.randint(0, len(population))] + mutation_vector
            else:
                new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]