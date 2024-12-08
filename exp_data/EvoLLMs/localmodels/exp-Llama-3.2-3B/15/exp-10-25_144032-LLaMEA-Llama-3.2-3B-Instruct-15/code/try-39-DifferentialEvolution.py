import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.best_individual = np.zeros(self.dim)
        self.best_fitness = -np.inf
        self.adaptive_mutation = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness = func(self.population)

            # Select the fittest individuals
            parents = np.argsort(self.fitness)[:-int(self.population_size/2):-1]
            self.population = self.population[parents]

            # Perform crossover and mutation
            for i in range(self.population_size):
                # Differential evolution
                differential = self.population[np.random.choice(parents, 3, replace=False)] - self.population[i]
                mutated_individual = self.population[i] + differential * np.random.uniform(0.5, 1.5, self.dim)
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

                # Adaptive mutation
                self.adaptive_mutation[i] = np.random.uniform(0.1, 0.3, self.dim)
                self.adaptive_mutation[i] = np.clip(self.adaptive_mutation[i], 0.1, 0.3)
                mutated_individual += self.adaptive_mutation[i]

                # Update the population and the best individual
                self.population = np.concatenate((self.population, mutated_individual.reshape(1, self.dim)))
                self.fitness = func(self.population)
                self.population = self.population[np.argsort(self.fitness)]
                self.best_individual = self.population[0]
                self.best_fitness = func(self.best_individual)

                # Check for convergence
                if self.best_fitness > self.best_fitness:
                    break

        return self.best_individual, self.best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = DifferentialEvolution(budget, dim)
best_individual, best_fitness = optimizer(func)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)