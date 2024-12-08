import numpy as np

class DynamicMutationEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.mutation_rates = np.random.uniform(0, 1, self.population_size)

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget // self.population_size):
            population = [np.clip(np.random.normal(best_solution, self.mutation_rates[i]), -5.0, 5.0) for i in range(self.population_size)]
            fitness_values = [func(individual) for individual in population]

            for idx, fitness in enumerate(fitness_values):
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = population[idx]

            for i in range(self.population_size):
                ind1, ind2, ind3 = np.random.choice(range(self.population_size), 3, replace=False)
                mutant = best_solution + 0.5 * (population[ind1] - population[ind2])
                trial = np.where(np.random.uniform(0, 1, self.dim) < 0.5, mutant, best_solution)
                trial_fitness = func(trial)
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            self.mutation_rates = np.clip(np.random.normal(self.mutation_rates, 0.1), 0, 1)  # Adapt mutation rates
        
        return best_solution
