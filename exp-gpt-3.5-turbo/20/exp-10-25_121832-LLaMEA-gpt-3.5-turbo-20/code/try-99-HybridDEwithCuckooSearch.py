import numpy as np

class HybridDEwithCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cuckoo_prob = 0.2
        self.de_crossover_prob = 0.9

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        population = initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for i, x in enumerate(population):
                cuckoo_individual = np.random.uniform(-5.0, 5.0, self.dim)
                if func(cuckoo_individual) < func(x) and np.random.rand() < self.cuckoo_prob:
                    population[i] = cuckoo_individual

                # Perform DE mutation and crossover
                r1, r2, r3 = np.random.choice(range(self.population_size), 3, replace=False)
                mutant = population[r1] + 0.5 * (population[r2] - population[r3])

                trial = np.where(np.random.rand(self.dim) < self.de_crossover_prob, mutant, x)
                if func(trial) < func(x):
                    population[i] = trial

                evaluations += 1
                if evaluations >= self.budget:
                    break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution