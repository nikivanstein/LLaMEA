import numpy as np

class ProbEvolutionary:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.mutation_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the objective function
            values = func(self.population)

            # Calculate the fitness of each individual
            fitness = np.sum(values, axis=1)

            # Select the fittest individuals
            indices = np.argsort(fitness)[-self.population_size // 2:]
            self.population = self.population[indices]

            # Perform mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    self.population[i] += np.random.uniform(-0.1, 0.1, self.dim)

            # Perform crossover
            for i in range(self.population_size // 2):
                parent1, parent2 = np.random.choice(self.population_size, 2, replace=False)
                child = (self.population[parent1] + self.population[parent2]) / 2
                self.population[i * 2], self.population[i * 2 + 1] = child, self.population[parent1]

# Example usage
if __name__ == "__main__":
    budget = 100
    dim = 10
    optimizer = ProbEvolutionary(budget, dim)
    func = lambda x: np.sum(x**2)  # Example function
    optimizer(func)