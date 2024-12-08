import numpy as np

class DynamicMutationStrengthDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutation(self, target, population, fitness):
        mutation_strength = np.clip(np.mean(fitness) - fitness, 0.1, 1.0)
        mutant = target + mutation_strength * self.CR * (population[np.random.randint(self.pop_size)] - target)
        return mutant

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        fitness = [func(ind) for ind in population]

        for _ in range(self.budget):
            new_population = []
            for target in population:
                a, b, c = population[np.random.choice(range(self.pop_size), 3, replace=False)]
                trial = self.mutation(target, population, fitness)
                mask = np.random.rand(self.dim) < self.CR
                offspring = np.where(mask, trial, target)
                new_population.append(offspring)

            population = np.array(new_population)
            fitness = [func(ind) for ind in population]

        return population[np.argmin([func(ind) for ind in population])]