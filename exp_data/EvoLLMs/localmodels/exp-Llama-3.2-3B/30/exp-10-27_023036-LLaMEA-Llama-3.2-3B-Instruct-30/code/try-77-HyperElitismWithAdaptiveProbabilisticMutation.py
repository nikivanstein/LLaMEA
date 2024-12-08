import numpy as np

class HyperElitismWithAdaptiveProbabilisticMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.fitness_values = np.zeros(self.budget)

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        return population

    def evaluate(self, func):
        for i, x in enumerate(self.population):
            self.fitness_values[i] = func(x)
        self.population = self.select_elites()
        self.population = self.mutate(self.population)
        self.population = self.crossover(self.population)

    def select_elites(self):
        elite_size = int(self.budget * 0.2)
        sorted_indices = np.argsort(self.fitness_values)
        return self.population[sorted_indices[:elite_size]]

    def mutate(self, population):
        mutation_rate = 0.1
        mutated_population = population.copy()
        for i in range(self.budget):
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(0, self.dim)
                mutated_population[i, idx] += np.random.uniform(-1.0, 1.0)
                mutated_population[i, idx] = np.clip(mutated_population[i, idx], -5.0, 5.0)
        return mutated_population

    def crossover(self, population):
        crossover_rate = 0.5
        crossover_population = population.copy()
        for i in range(self.budget):
            if np.random.rand() < crossover_rate:
                idx1 = np.random.randint(0, self.dim)
                idx2 = np.random.randint(0, self.dim)
                while idx1 == idx2:
                    idx2 = np.random.randint(0, self.dim)
                crossover_population[i, idx1], crossover_population[i, idx2] = crossover_population[i, idx2], crossover_population[i, idx1]
        return crossover_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)

def optimize_function(func, budget, dim):
    algorithm = HyperElitismWithAdaptiveProbabilisticMutation(budget, dim)
    algorithm()
    return func(algorithm.population[0])

# Example usage
def noiseless_function(x):
    return np.sum(x**2)

budget = 100
dim = 10
func = noiseless_function
result = optimize_function(func, budget, dim)
print(result)