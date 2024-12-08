import numpy as np
from scipy.optimize import differential_evolution

class DEBOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the DEBOptimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None

    def __call__(self, func):
        """
        Optimize a black box function using DEBOptimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized function and its value.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize the population size and the number of generations
        population_size = 100
        num_generations = 100

        # Initialize the population with random solutions
        self.population = [np.random.uniform(lower_bound, upper_bound, size=(population_size, self.dim)) for _ in range(population_size)]

        # Evaluate the objective function for each individual in the population
        results = []
        for _ in range(num_generations):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Define the mutation function
        def mutate(individual):
            if np.random.rand() < 0.25:
                # Randomly select a new individual from the search space
                new_individual = np.random.uniform(lower_bound, upper_bound, size=self.dim)
                # Update the new individual with the mutation strategy
                new_individual[individual] += np.random.uniform(-1, 1)
                return new_individual
            else:
                return individual

        # Define the selection function
        def select(fittest_individuals, fitness_values):
            # Select the fittest individuals with a probability of 0.75
            selected_individuals = fittest_individuals[:len(fittest_individuals)//2]
            # Select the remaining individuals with a probability of 0.25
            remaining_individuals = fittest_individuals[len(fittest_individuals)//2:]
            # Return the selected individuals
            return selected_individuals + remaining_individuals

        # Update the population with the fittest individuals
        self.population = select(results, fitness_values)

        # Evaluate the objective function for the new population
        new_individual, _ = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

        # Return the optimized function and its value
        return func(new_individual), -func(new_individual)

# Example usage:
# optimizer = DEBOptimizer(100, 5)
# optimized_function, optimized_value = optimizer(func)
# print(f"Optimized function: {optimized_function}, Optimized value: {optimized_value}")