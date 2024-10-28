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

    def refine_strategy(self, func, func_values, population):
        # Calculate the average function value
        avg_func_value = np.mean(func_values)

        # Select the fittest solutions with the highest average function value
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]

        # Evolve the population using evolutionary differential evolution with refined strategy
        for _ in range(self.budget):
            # Select parents using tournament selection with refined strategy
            parents = np.array([self.population[i] for i in fittest_indices])

            # Perform mutation
            mutated_parents = parents.copy()
            for _ in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutated_parents[_] += np.random.normal(0, 1, self.dim)

            # Select offspring using tournament selection with refined strategy
            offspring = np.array([self.population[i] for i in np.argsort(mutated_parents)[::-1][:self.population_size]])

            # Replace the old population with the new one
            self.population = np.concatenate((self.population, mutated_parents), axis=0)
            self.population = np.concatenate((self.population, offspring), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

# One-line description with the main idea
# EvoDiff: An evolutionary differential evolution algorithm that leverages the concept of evolutionary differences to optimize black box functions with a refining strategy.

# Example usage
if __name__ == "__main__":
    # Create an instance of EvoDiff with a budget of 1000 evaluations
    evodiff = EvoDiff(budget=1000, dim=5)

    # Define a black box function
    def func(x):
        return np.sin(x)

    # Initialize the population with random solutions
    population = evodiff.init_population()

    # Evaluate the function with the initial population
    func_values = np.array([func(x) for x in population])

    # Optimize the function using EvoDiff
    optimized_func_values = evodiff(population, func_values)

    # Print the optimized function values
    print("Optimized function values:", optimized_func_values)