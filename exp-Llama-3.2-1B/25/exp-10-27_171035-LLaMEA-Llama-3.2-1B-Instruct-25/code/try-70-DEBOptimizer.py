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

        # Refine the solution using line search
        self.population = self.refine_solution(self.population)

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

    def refine_solution(self, population):
        """
        Refine the solution using line search.

        Args:
            population (list): The population of individuals.

        Returns:
            list: The refined population.
        """
        # Initialize the best individual and its value
        best_individual = population[0]
        best_value = -func(best_individual)

        # Initialize the step size and the number of iterations
        step_size = 0.1
        num_iterations = 10

        # Iterate over the population to find the best individual
        for i in range(num_iterations):
            # Evaluate the objective function at the current step
            fitness_values = differential_evolution(lambda x: -func(x), [best_individual] * 10, bounds=(lower_bound, upper_bound), x0=best_individual)

            # Update the best individual and its value
            best_individual = [x for x, _ in zip(population, fitness_values.x) if _ == fitness_values.x[0]][0]
            best_value = -func(best_individual)

            # Update the step size
            step_size *= 0.9

        # Update the population with the best individual
        self.population = [best_individual] * 10 + population[10:]

        return self.population