import numpy as np
import concurrent.futures

class HybridGA_PSO_Parallel:
    def __init__(self, budget, dim, population_size=50, mutation_rate=0.1, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._update_individual, ind, func) for ind in population]
                results = [future.result() for future in futures]
                population = np.array(results)
        
        return population[np.argmin([func(ind) for ind in population])]

    def _update_individual(self, individual, func):
        # Update velocity based on PSO
        velocity = self.inertia_weight * np.random.rand() + self.cognitive_weight * np.random.rand() * (best_individual - individual) + self.social_weight * np.random.rand() * (global_best - individual)
        individual += velocity
        
        # Mutate based on GA
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate
        individual = individual + np.random.uniform(-1.0, 1.0, self.dim) * mutation_mask
        
        return individual