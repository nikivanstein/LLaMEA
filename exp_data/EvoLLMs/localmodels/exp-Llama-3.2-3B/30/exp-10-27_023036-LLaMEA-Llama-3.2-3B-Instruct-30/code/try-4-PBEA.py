import numpy as np

class PBEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.fitness = self.evaluate_fitness()

    def initialize_population(self):
        population = []
        for _ in range(self.budget):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(x)
        return population

    def evaluate_fitness(self):
        fitness = np.zeros(self.budget)
        for i, x in enumerate(self.population):
            fitness[i] = func(x)
        return fitness

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents using probability-based selection
            parents = np.random.choice(self.population, size=self.dim, replace=False, p=self.fitness / np.sum(self.fitness))
            # Perform crossover and mutation
            offspring = []
            for _ in range(self.dim):
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                child = (parent1 + parent2) / 2 + np.random.uniform(-0.5, 0.5, self.dim)
                offspring.append(child)
            # Evaluate fitness of offspring
            fitness_offspring = np.zeros(self.dim)
            for i, x in enumerate(offspring):
                fitness_offspring[i] = func(x)
            # Update population and fitness
            self.population = np.concatenate((self.population, offspring))
            self.fitness = np.concatenate((self.fitness, fitness_offspring))
            # Replace worst individual with the best one
            self.population = self.population[np.argsort(self.fitness)]
            self.fitness = self.fitness[np.argsort(self.fitness)]

# Example usage:
budget = 100
dim = 10
pbea = PBEA(budget, dim)
pbea(func)