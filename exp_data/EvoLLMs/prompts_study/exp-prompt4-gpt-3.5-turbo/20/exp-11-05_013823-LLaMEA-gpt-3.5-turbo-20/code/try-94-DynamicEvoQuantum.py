import numpy as np

class DynamicEvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.elite_ratio = 0.1
        self.mutation_rate = 0.1  # Initialize mutation rate
        
    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)
            elite_count = int(self.budget * self.elite_ratio)
            elite = self.population[sorted_indices[:elite_count]]  # Select top elite_ratio as elite
            
            # Adaptive mutation rate based on individual performance
            best_fitness = func(elite[0])
            worst_fitness = func(elite[-1])
            performance_ratio = (worst_fitness - fitness_values) / (worst_fitness - best_fitness + 1e-8)
            mutation_rate = 0.1 / (1 + np.mean(performance_ratio))  # Adjust mutation rate based on individual performance
            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - mutation_rate, mutation_rate])
            new_population = elite + mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))
            
            self.population = new_population
        best_solution = elite[0]  # Select the best solution from the elite
        return func(best_solution)