import numpy as np
from joblib import Parallel, delayed

class HybridGA_PSO:
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
            # Update velocity based on PSO in parallel
            velocity = Parallel(n_jobs=-1)(
                delayed(self.update_velocity)(ind, population, fitness) for ind in population
            )
            
            # Mutate based on GA in parallel
            mutated_population = Parallel(n_jobs=-1)(
                delayed(self.mutate_individual)(ind) for ind in population
            )
            
            population = np.array(mutated_population)
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]
        
        return global_best

    def update_velocity(self, ind, population, fitness):
        return self.inertia_weight * ind + self.cognitive_weight * np.random.rand() * (population[np.argmin(fitness)] - ind) + self.social_weight * np.random.rand() * (population[np.argmin(fitness)] - ind)

    def mutate_individual(self, ind):
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate
        return ind + np.random.uniform(-1.0, 1.0, self.dim) * mutation_mask