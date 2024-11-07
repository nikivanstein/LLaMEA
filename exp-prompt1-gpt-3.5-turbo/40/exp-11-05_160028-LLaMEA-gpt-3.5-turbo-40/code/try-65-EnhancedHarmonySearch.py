import numpy as np

class EnhancedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim, local_search_prob=0.1):
        super().__init__(budget, dim)
        self.local_search_prob = local_search_prob

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_fitness = func(new_harmony)
            
            if np.random.rand() < self.local_search_prob:
                local_search_point = population[np.random.randint(self.budget)]
                new_harmony = (new_harmony + local_search_point) / 2
                new_fitness = func(new_harmony)

            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]