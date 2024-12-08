import numpy as np

class ProbabilisticAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (np.array([-5.0, 5.0]) + 1) / 2
        self.population_size = 20
        self.population = np.random.uniform(self.search_space, size=(self.population_size, self.dim))
        self.fitness_values = np.zeros((self.population_size,))

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate fitness for each individual
            self.fitness_values = func(self.population)
            
            # Select parents using tournament selection
            parents = np.random.choice(self.population_size, size=self.population_size, replace=False, p=self.fitness_values / np.sum(self.fitness_values))
            
            # Crossover (recombination)
            offspring = np.zeros((self.population_size, self.dim))
            for j in range(self.population_size):
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                offspring[j] = (self.population[parent1] + self.population[parent2]) / 2
            
            # Adaptive mutation
            mutation_probabilities = np.random.uniform(0, 1, size=self.population_size)
            for j in range(self.population_size):
                if mutation_probabilities[j] < 0.3:
                    mutation = np.random.uniform(-1, 1, size=self.dim)
                    offspring[j] += mutation
            
            # Replace worst individual
            self.population = np.minimum(self.population, offspring)
            
            # Print progress
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best Fitness: {np.min(self.fitness_values)}")

# Example usage
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = ProbabilisticAdaptiveMutation(budget, dim)
best_fitness = np.inf
for i in range(budget):
    optimizer(func)
    if np.min(optimizer.fitness_values) < best_fitness:
        best_fitness = np.min(optimizer.fitness_values)
        print(f"Best Fitness: {best_fitness}")