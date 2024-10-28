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

    def refine_strategy(self, func, new_individual, new_func_values):
        # Calculate the fitness difference between the new individual and the old population
        fitness_diff = new_func_values - func(new_individual)

        # Refine the strategy by changing the mutation rate and the number of iterations
        self.mutation_rate = 0.01 + 0.5 * np.random.uniform(-1, 1)
        self.budget = 100 + 10 * np.random.uniform(-1, 1)

# Usage
problem = ioh.iohcpp.problem.RealSingleObjective()
problem.set_problem("Sphere", iid=1, dim=5)
problem.set_bounds(-5.0, 5.0)

evo_diff = EvoDiff(budget=100, dim=5, population_size=100, mutation_rate=0.01)
evo_diff.init_population()

# Optimize the function using EvoDiff
new_individual = evo_diff.evaluate_fitness(problem.evaluate_func)
new_func_values = problem.evaluate_func(new_individual)

# Refine the strategy
evo_diff.refine_strategy(problem, new_individual, new_func_values)

# Optimize the function again
new_individual = evo_diff.evaluate_fitness(problem.evaluate_func)
new_func_values = problem.evaluate_func(new_individual)

# Print the final fitness values
print("Final Fitness Values:")
print(problem.evaluate_func(new_individual))