import numpy as np
import random
import os

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.population_size = 100
        self.population_history = []
        self.budgets = []
        self.noise_levels = []

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-gradient descent.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

        # Store the optimized parameter values and the objective function value
        self.population_history.append((self.param_values, func_value))
        self.budgets.append(_)
        self.noise_levels.append(self.noise_level)

        # Refine the solution by changing the individual lines of the selected solution
        self.refine_solution()

    def refine_solution(self):
        """
        Refine the solution by changing the individual lines of the selected solution.
        """
        # Get the current population history
        current_population_history = self.population_history

        # Get the current budgets and noise levels
        current_budgets = self.budgets
        current_noise_levels = self.noise_levels

        # Select the individual with the best fitness value
        best_individual = current_population_history[0]

        # Select the individual with the highest fitness value
        best_individual_fitness = current_population_history[0][1]

        # Refine the selected individual
        for i in range(len(current_population_history)):
            for j in range(len(current_population_history[i])):
                # Create a copy of the current individual
                individual = current_population_history[i][0]

                # Change the individual line by line
                for k in range(self.dim):
                    individual[k] += random.uniform(-1, 1)

                # Evaluate the new individual
                new_individual, new_fitness = self.__call__(individual)

                # Update the best individual if the new individual has a better fitness value
                if new_fitness > best_individual_fitness:
                    best_individual = individual
                    best_individual_fitness = new_fitness

        # Store the refined solution
        self.population_history.append((best_individual, best_individual_fitness))

    def save_population_history(self, file_path):
        """
        Save the population history to a file.

        Args:
            file_path (str): The path to the file to save the population history.
        """
        with open(file_path, "wb") as file:
            for i in range(len(self.population_history)):
                np.save(file, self.population_history[i])

# Example usage
if __name__ == "__main__":
    meta_gradient_descent = MetaGradientDescent(budget=100, dim=10, noise_level=0.1)

    # Optimize a function
    def func(x):
        return x**2 + 2*x + 1

    meta_gradient_descent(func)

    # Save the population history
    meta_gradient_descent.save_population_history("population_history.npy")

    # Load the population history
    loaded_population_history = np.load("population_history.npy")

    # Print the optimized parameter values and the objective function value
    print("Optimized parameter values:", loaded_population_history[0][0])
    print("Objective function value:", loaded_population_history[0][1])