import numpy as np
import random
import copy

class MetaHeuristic:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-heuristic algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.population = []

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-heuristic algorithms.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        self.population = [copy.deepcopy(func(self.param_values)) for self.param_values in np.random.uniform(-5.0, 5.0, self.dim * self.dim)]

        # Evolve the population using a combination of exploration and exploitation strategies
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.population[0])

            # Select the individual with the best fitness
            best_individual = self.select_best_individual(func_value)

            # Explore the search space by generating new individuals
            new_individuals = self.explore_search_space(best_individual, func, self.population, self.dim)

            # Update the population with the new individuals
            self.population = self.update_population(self.population, new_individuals)

            # Update the best individual
            self.population[0] = best_individual

        # Return the optimized parameter values and the objective function value
        return self.population[0], func(self.population[0])

    def select_best_individual(self, func_value):
        """
        Select the individual with the best fitness.

        Args:
            func_value (float): The value of the objective function.

        Returns:
            int: The index of the best individual.
        """
        # Calculate the fitness ratio
        fitness_ratio = np.random.uniform(0, 1)

        # Select the individual with the best fitness
        best_individual = np.argmax(np.random.rand(len(self.population)) * fitness_ratio)
        return best_individual

    def explore_search_space(self, best_individual, func, population, dim):
        """
        Explore the search space by generating new individuals.

        Args:
            best_individual (int): The index of the best individual.
            func (callable): The black box function.
            population (list): The current population.
            dim (int): The dimensionality.

        Returns:
            list: A list of new individuals.
        """
        # Initialize the new population
        new_population = []

        # Generate new individuals by perturbing the best individual
        for _ in range(100):
            # Perturb the best individual
            perturbed_individual = copy.deepcopy(best_individual)
            perturbed_individual += np.random.normal(0, 1, dim)

            # Evaluate the new individual
            new_individual_value = func(perturbed_individual)

            # Add the new individual to the new population
            new_population.append(new_individual_value)

        # Return the new population
        return new_population

    def update_population(self, population, new_population):
        """
        Update the population with the new individuals.

        Args:
            population (list): The current population.
            new_population (list): The new population.

        Returns:
            list: The updated population.
        """
        # Combine the old and new populations
        updated_population = population + new_population

        # Return the updated population
        return updated_population

# Example usage:
if __name__ == "__main__":
    # Create a new meta-heuristic algorithm
    meta_heuristic = MetaHeuristic(100, 10)

    # Optimize the function f(x) = x^2
    func = lambda x: x**2
    best_individual, func_value = meta_heuristic(func)

    # Print the result
    print("Best individual:", best_individual)
    print("Objective function value:", func_value)