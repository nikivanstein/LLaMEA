import numpy as np

class CRP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.fitness_values = np.zeros(self.population_size)

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def fitness(self, individual):
        return func(individual)

    def evaluate(self):
        self.fitness_values = np.array([self.fitness(individual) for individual in self.population])

    def selection(self):
        indices = np.argsort(self.fitness_values)
        selected_indices = indices[:int(self.population_size/2)]
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                child[i] = parent2[i]
        return child

    def recombination(self):
        children = []
        while len(children) < self.population_size - self.population_size // 2:
            parent1, parent2 = np.random.choice(self.population, size=2, replace=False)
            child = self.crossover(parent1, parent2)
            children.append(child)
        return children

    def mutation(self, individual):
        mutation_rate = 0.1
        for i in range(self.dim):
            if np.random.rand() < mutation_rate:
                individual[i] += np.random.uniform(-1.0, 1.0)
                individual[i] = np.clip(individual[i], -5.0, 5.0)
        return individual

    def update(self):
        self.evaluate()
        selected_individuals = self.selection()
        children = self.recombination()
        for i, child in enumerate(children):
            parent1, parent2 = selected_individuals[i % len(selected_individuals)]
            child = self.mutation(child)
            child = self.crossover(parent1, child)
            self.population[i] = child
        return np.mean(self.fitness_values)

    def __call__(self, func):
        for _ in range(self.budget):
            self.update()
        return np.mean(self.fitness_values)

# Example usage:
func = lambda x: np.sum(x**2)
crp = CRP(budget=100, dim=10)
result = crp(func)
print(result)