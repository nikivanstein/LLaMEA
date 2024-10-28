import numpy as np

class EvoDiff:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        return np.random.uniform(-5.0, 5.0, self.dim) + np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        # Evaluate the black box function with the current population
        func_values = np.array([func(x) for x in self.population])

        # Select the fittest solutions
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]

        # Evolve the population using evolutionary differential evolution
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = np.array([self.population[i] for i in fittest_indices])

            # Perform mutation
            mutated_parents = parents.copy()
            for _ in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutated_parents[_] += np.random.normal(0, 1, self.dim)

            # Select offspring using tournament selection
            offspring = np.array([self.population[i] for i in np.argsort(mutated_parents)[::-1][:self.population_size]])

            # Replace the old population with the new one
            self.population = np.concatenate((self.population, mutated_parents), axis=0)
            self.population = np.concatenate((self.population, offspring), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

    def select_parents(self, func_values, func):
        # Select parents using tournament selection
        parents = np.array([func(x) for x in self.population])
        return np.array([self.population[i] for i in np.argsort(parents)[::-1][:self.population_size]])

    def mutate(self, parents, func_values):
        # Perform mutation
        mutated_parents = parents.copy()
        for _ in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutated_parents[_] += np.random.normal(0, 1, self.dim)
        return mutated_parents

# Example usage:
def func(x):
    return np.sum(x**2)

evolution_diff = EvoDiff(budget=100, dim=5)
evolution_diff.population = np.random.uniform(-5.0, 5.0, 5) + np.random.normal(0, 1, 5)
print(evolution_diff(func))

parents = evolution_diff.select_parents(func_values=func(np.random.uniform(-5.0, 5.0, 5)), func=func)
mutated_parents = evolution_diff.mutate(parents, func_values=func)

new_individual = evolution_diff.func(np.concatenate((evolution_diff.population, mutated_parents), axis=0))
print(new_individual)