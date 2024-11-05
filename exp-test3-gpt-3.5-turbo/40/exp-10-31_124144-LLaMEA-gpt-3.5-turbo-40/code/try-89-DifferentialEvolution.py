import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, scale_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.scale_factor = scale_factor
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        best_solution = population[np.argmin(fitness)]
        for _ in range(self.budget):
            rand_1, rand_2, rand_3 = np.random.choice(self.budget, 3, replace=False)
            mutant = population[rand_1] + self.scale_factor * np.random.standard_cauchy(self.dim) * (population[rand_2] - population[rand_3])
            mutant_fitness = func(mutant)
            if mutant_fitness < fitness[rand_1]:
                population[rand_1] = mutant
                fitness[rand_1] = mutant_fitness
                if mutant_fitness < func(best_solution):
                    best_solution = mutant
        return best_solution