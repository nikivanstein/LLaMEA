import numpy as np

class AdaptiveBiogeographyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = min(max(5 * dim, 30), budget // 3)
        self.mutation_rate = 0.3
        self.elitism_rate = 0.2
        self.migration_rate = 0.8
        self.best_individual = None
        self.best_value = float('inf')
    
    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
    
    def evaluate_population(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_value:
                self.best_value = self.fitness[i]
                self.best_individual = self.population[i].copy()
    
    def migrate(self):
        sorted_indices = np.argsort(self.fitness)
        elites = sorted_indices[:int(self.elitism_rate * self.population_size)]
        for i in range(self.population_size):
            if np.random.rand() < self.migration_rate:
                donor_idx = np.random.choice(elites)
                donor = self.population[donor_idx]
                mask = np.random.rand(self.dim) < self.migration_rate
                self.population[i][mask] = donor[mask]
    
    def mutate(self):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.uniform(self.lb, self.ub, self.dim)
                mutation_mask = np.random.rand(self.dim) < self.mutation_rate
                self.population[i][mutation_mask] = mutation_vector[mutation_mask]
    
    def __call__(self, func):
        self.initialize_population()
        evals = 0
        while evals < self.budget:
            self.evaluate_population(func)
            self.migrate()
            self.mutate()
            evals += self.population_size
        return self.best_individual, self.best_value