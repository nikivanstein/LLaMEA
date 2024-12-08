import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.probability_refinement = 0.35

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        population_size = 10
        mutation_rates = np.random.uniform(0, 1, population_size)

        for _ in range(self.budget // population_size):
            population = [np.clip(np.random.normal(best_solution, mutation_rates[i]), -5.0, 5.0) for i in range(population_size)]
            fitness_values = [func(individual) for individual in population]

            for idx, fitness in enumerate(fitness_values):
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = population[idx]

            for i in range(population_size):
                ind1, ind2, ind3 = np.random.choice(range(population_size), 3, replace=False)
                mutant = best_solution + 0.5 * (population[ind1] - population[ind2])
                trial = np.where(np.random.uniform(0, 1, self.dim) < 0.5, mutant, best_solution)
                trial_fitness = func(trial)
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            mutation_rates = np.clip(np.random.normal(mutation_rates, 0.1), 0, 1)  # Adapt mutation rates

            if np.random.uniform(0, 1) < self.probability_refinement:
                population_size += 1
                mutation_rates = np.append(mutation_rates, np.random.uniform(0, 1))
        
        return best_solution