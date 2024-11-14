import numpy as np

class ImprovedAdaptiveMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_probs = np.ones(dim) * 0.5

    def levy_flight(self, scale=1.0):
        return np.random.standard_cauchy() * scale / (np.abs(np.random.normal()) ** (1 / self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]

            new_population = []
            for individual in self.population:
                step = self.levy_flight()
                mutant = individual + step
                new_population.append(mutant)

            self.population = np.array(new_population)
            self.mutation_probs *= 0.95  # Adapt mutation probabilities

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual