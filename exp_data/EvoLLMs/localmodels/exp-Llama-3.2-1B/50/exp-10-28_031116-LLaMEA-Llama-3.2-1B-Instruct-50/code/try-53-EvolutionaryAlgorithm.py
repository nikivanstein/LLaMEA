import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness_scores = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def fitness(individual):
            func(individual)
            return self.fitness_scores[individual]

        for _ in range(self.budget):
            individual = np.random.choice(self.population_size, size=self.dim, replace=False)
            fitness(individual)

        best_individual = np.argmax(self.fitness_scores)
        return best_individual, fitness(best_individual)

    def mutate(self, individual):
        if np.random.rand() < 0.45:
            individual = np.random.uniform(individual.min(), individual.max() + 1)
        return individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.45:
            child = np.random.uniform(parent1.min(), parent1.max())
            child[parent1.shape[0]:] = parent2[parent2.shape[0]:]
            return child
        else:
            child = np.concatenate((parent1[:parent1.shape[0]//2], parent2[parent2.shape[0]//2:]))
            return child