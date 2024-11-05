import numpy as np

class AdaptiveDynamicDE:
    def __init__(self, budget, dim, mutation_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        diversity = np.std(population)
        
        for _ in range(self.budget):
            F = np.random.uniform(0.1, 0.9)
            mutant = self.mutation(population, F, diversity)
            trial = np.where(np.random.uniform(0, 1, self.dim) < self.mutation_rate, mutant, population)
            trial_fitness = func(trial)
            if trial_fitness < fitness[best_idx]:
                best_solution = trial
                fitness[best_idx] = trial_fitness
            population, fitness = self.selection(population, fitness, trial, trial_fitness)

        return best_solution

    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * diversity
        return np.clip(mutant, -5.0, 5.0)

    def selection(self, population, fitness, trial, trial_fitness):
        replace_idx = np.where(trial_fitness < fitness)[0]
        population[replace_idx] = trial[replace_idx]
        fitness[replace_idx] = trial_fitness[replace_idx]
        return population, fitness