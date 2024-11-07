import numpy as np

class DynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_rate = 0.1
        self.memory = []

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
                if len(self.memory) < int(self.memory_rate * self.budget):
                    self.memory.append(new_harmony)
                else:
                    self.memory.pop(0)
                    self.memory.append(new_harmony)
            
            for mem_harmony in self.memory:
                mem_fitness = func(mem_harmony)
                if mem_fitness < fitness[worst_idx]:
                    population[worst_idx] = mem_harmony
                    fitness[worst_idx] = mem_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]