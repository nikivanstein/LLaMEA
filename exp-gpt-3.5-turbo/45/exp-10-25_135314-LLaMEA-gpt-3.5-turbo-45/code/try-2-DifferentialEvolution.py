import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR

    def mutate(self, population, current_index):
        candidates = [idx for idx in range(self.budget) if idx != current_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        return population[a] + self.F * (population[b] - population[c])

    def crossover(self, target, trial):
        crossover_points = np.random.rand(self.dim) < self.CR
        offspring = np.where(crossover_points, trial, target)
        return offspring

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness_values = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            for i in range(self.budget):
                trial = self.mutate(population, i)
                offspring = self.crossover(population[i], trial)
                trial_fitness = func(trial)
                if trial_fitness < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness

        return population[np.argmin(fitness_values)]