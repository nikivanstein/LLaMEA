import numpy as np
import random
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

    def genetic_algorithm(self, func, initial_population, mutation_rate, cooling_rate, num_generations):
        """
        Perform a genetic algorithm to optimize the black box function.

        Args:
            func (callable): The black box function to optimize.
            initial_population (list): The initial population of individuals.
            mutation_rate (float): The probability of mutation.
            cooling_rate (float): The rate at which the temperature decreases.
            num_generations (int): The number of generations.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the population with random individuals
        population = initial_population

        # Perform the specified number of generations
        for _ in range(num_generations):
            # Evaluate the fitness of each individual
            fitness = [self.func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = [individual for _, individual in sorted(zip(fitness, population), reverse=True)]

            # Create a new population by mutating the fittest individuals
            new_population = []
            for _ in range(len(fittest_individuals)):
                parent1, parent2 = fittest_individuals[np.random.randint(0, len(fittest_individuals) - 1)], fittest_individuals[np.random.randint(0, len(fittest_individuals) - 1)]
                child = parent1[:self.dim] + [random.uniform(-5.0, 5.0) for _ in range(self.dim)] + [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
                new_population.append(child)

            # Evaluate the fitness of the new population
            new_fitness = [self.func(individual) for individual in new_population]

            # Select the fittest individuals from the new population
            fittest_new_individuals = [individual for _, individual in sorted(zip(new_fitness, new_population), reverse=True)]

            # Create a new population by combining the fittest individuals
            new_population = fittest_new_individuals[:self.dim] + fittest_new_individuals[self.dim:]

            # If the temperature is below a certain threshold, stop the algorithm
            if self.temperature < 0.01:
                break

            # Update the temperature
            self.temperature *= cooling_rate

            # Add the new population to the population
            population.extend(new_population)

        # Return the fittest individual in the final population
        return population[0]

    def simulated_annealing(self, func, initial_temperature, cooling_rate, num_iterations):
        """
        Perform a simulated annealing algorithm to optimize the black box function.

        Args:
            func (callable): The black box function to optimize.
            initial_temperature (float): The initial temperature.
            cooling_rate (float): The rate at which the temperature decreases.
            num_iterations (int): The number of iterations.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the current temperature
        current_temperature = initial_temperature

        # Perform the specified number of iterations
        for _ in range(num_iterations):
            # Evaluate the fitness of each individual
            fitness = [self.func(individual) for individual in self.search_space]

            # Select the fittest individuals
            fittest_individuals = [individual for _, individual in sorted(zip(fitness, self.search_space), reverse=True)]

            # Create a new population by mutating the fittest individuals
            new_population = []
            for _ in range(len(fittest_individuals)):
                parent1, parent2 = fittest_individuals[np.random.randint(0, len(fittest_individuals) - 1)], fittest_individuals[np.random.randint(0, len(fittest_individuals) - 1)]
                child = parent1[:self.dim] + [random.uniform(-5.0, 5.0) for _ in range(self.dim)] + [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
                new_population.append(child)

            # Evaluate the fitness of the new population
            new_fitness = [self.func(individual) for individual in new_population]

            # Select the fittest individuals from the new population
            fittest_new_individuals = [individual for _, individual in sorted(zip(new_fitness, new_population), reverse=True)]

            # Create a new population by combining the fittest individuals
            new_population = fittest_new_individuals[:self.dim] + fittest_new_individuals[self.dim:]

            # If the current temperature is below a certain threshold, stop the algorithm
            if current_temperature < 0.01:
                break

            # Update the current temperature
            current_temperature *= cooling_rate

        # Return the fittest individual in the final population
        return self.search_space[np.argmax([self.func(individual) for individual in self.search_space])]

# Example usage:
budget = 100
dim = 5
func = lambda x: x**2 + 2*x + 1
optimizer = BlackBoxOptimizer(budget, dim)
optimized_value = optimizer(func)

# Print the optimized value
print(f"Optimized value: {optimized_value}")