import numpy as np

class DynamicDifferentialEvolution:
    def __init__(self, budget, dim, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.mu, self.dim))

        def mutate_population(population):
            mutated_population = np.zeros((self.lambda_, self.dim))
            for i in range(self.lambda_):
                a, b, c = np.random.choice(range(self.lambda_), 3, replace=False)
                mutant = np.clip(population[a] + self.f * (population[b] - population[c]), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < self.cr
                offspring = np.where(crossover, mutant, population[i])
                mutated_population[i] = np.clip(offspring, -5.0, 5.0)
            return mutated_population

        population = initialize_population()
        for _ in range(self.budget):
            offspring = mutate_population(population)
            fitness = np.array([func(individual) for individual in offspring])
            best_index = np.argmin(fitness)
            if fitness[best_index] < func(population[0]):
                population[0] = offspring[best_index]
            self.f = np.clip(np.random.normal(self.f, 0.1), 0.1, 1.0)  # Adaptive mutation factor
            self.cr = np.clip(np.random.normal(self.cr, 0.1), 0.1, 1.0)  # Adaptive crossover rate
        return population[0]