import numpy as np

class AdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.selection_prob = 0.3
        self.crossover_prob = 0.5
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        for individual in self.population:
            func(individual)
        self.population = self.select_and_crossover()

    def select(self):
        scores = np.array([func(individual) for individual in self.population])
        probabilities = scores / np.max(scores)
        selected_indices = np.random.choice(len(self.population), size=self.population_size, p=probabilities)
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def crossover(self):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = np.random.choice(self.population, size=2, replace=False)
            child = self.adaptive_crossover(parent1, parent2)
            new_population.append(child)
        return new_population

    def adaptive_crossover(self, parent1, parent2):
        child = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.crossover_prob:
                if np.random.rand() < self.selection_prob:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            else:
                child[i] = (parent1[i] + parent2[i]) / 2
        if np.random.rand() < self.mutation_rate:
            child += np.random.uniform(-0.1, 0.1, self.dim)
        return child

    def mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.uniform(-0.1, 0.1)
        return individual

    def select_and_crossover(self):
        selected_individuals = self.select()
        new_population = self.crossover()
        for individual in selected_individuals:
            new_population.append(individual)
        new_population = [self.mutate(individual) for individual in new_population]
        return new_population

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
algorithm = AdaptiveCrossover(budget, dim)
for _ in range(budget):
    algorithm.evaluate(func)