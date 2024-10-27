import numpy as np
import random

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.

    The algorithm uses a combination of genetic algorithm and simulated annealing to find the optimal solution.
    """

    def __init__(self, budget, dim):
        """
        Initializes the optimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimizes a black box function using the optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the solution and the number of function evaluations
        solution = None
        evaluations = 0

        # Iterate over the range of possible solutions
        while evaluations < self.budget:
            # Generate a random solution within the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the black box function at the current solution
            evaluations += 1
            func(solution)

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        solution = solution
                else:
                    # Update the solution with the best solution found so far
                    solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations

    def simulated_annealing(self, func, initial_solution, temperature, cooling_rate):
        """
        Simulated annealing algorithm to optimize a black box function.

        Args:
            func (function): The black box function to optimize.
            initial_solution (tuple): The initial solution.
            temperature (float): The initial temperature.
            cooling_rate (float): The cooling rate.
        """
        # Initialize the current solution and the current temperature
        current_solution = initial_solution
        current_temperature = temperature

        # Iterate until the temperature is reduced to 0
        while current_temperature > 0.1:
            # Calculate the new solution using the current solution and a small random change
            new_solution = current_solution + np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the new solution
            new_evaluations = 1
            func(new_solution)

            # If the new solution is better than the current solution, accept it
            if new_evaluations > 0 and new_evaluations < self.budget:
                if new_evaluations > 0:
                    # Calculate the probability of accepting the new solution
                    probability = np.exp((new_evaluations - new_evaluations) / self.budget)

                    # Accept the new solution with a probability less than 1
                    if np.random.rand() < probability:
                        current_solution = new_solution
                        current_temperature *= cooling_rate
                else:
                    # Update the current solution with the new solution
                    current_solution = new_solution

            # If the new solution is not better than the current solution, accept it with a probability less than 1
            else:
                if np.random.rand() < np.exp((new_evaluations - new_evaluations) / self.budget):
                    current_solution = new_solution

        # Return the optimal solution and the number of function evaluations used
        return current_solution, new_evaluations

# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Genetic Algorithm for Black Box Optimization
def genetic_algorithm(func, population_size, mutation_rate, initial_population):
    """
    Genetic algorithm for black box optimization.

    Args:
        func (function): The black box function to optimize.
        population_size (int): The size of the population.
        mutation_rate (float): The mutation rate.
        initial_population (list): The initial population.

    Returns:
        tuple: A tuple containing the optimal solution and the number of function evaluations used.
    """
    # Initialize the population with random solutions
    population = initial_population

    # Iterate until the population is reduced to 1
    while len(population) > 1:
        # Evaluate the fitness of each solution in the population
        fitnesses = [func(solution) for solution in population]

        # Select the fittest solutions
        fittest_solutions = [solution for solution, fitness in zip(population, fitnesses) if fitness == max(fitnesses)]

        # Create a new population by mutating the fittest solutions
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(fittest_solutions, 2)
            child = parent1 + parent2 * mutation_rate
            new_population.append(child)

        # Evaluate the fitness of each solution in the new population
        fitnesses = [func(solution) for solution in new_population]

        # Select the fittest solutions
        fittest_solutions = [solution for solution, fitness in zip(new_population, fitnesses) if fitness == max(fitnesses)]

        # Replace the old population with the new population
        population = fittest_solutions

    # Return the optimal solution and the number of function evaluations used
    return population[0], len(population)

# Example usage:
def func(x):
    return x**2 + 2*x + 1

population_size = 100
mutation_rate = 0.1
initial_population = [random.uniform(-5.0, 5.0) for _ in range(population_size)]

optimal_solution, num_evaluations = genetic_algorithm(func, population_size, mutation_rate, initial_population)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)