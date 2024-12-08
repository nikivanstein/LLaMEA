import numpy as np
import random

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.
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

        # Define the population size and the mutation rate
        population_size = 100
        mutation_rate = 0.01

        # Initialize the population with random solutions
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(population_size)]

        # Define the fitness function to evaluate the solutions
        def fitness(individual):
            return np.linalg.norm(func(individual))

        # Evaluate the solutions for a given number of iterations
        for _ in range(self.budget):
            # Evaluate the fitness of each solution
            fitness_values = [fitness(individual) for individual in population]

            # Select the fittest solutions
            fittest_indices = np.argsort(fitness_values)[-self.dim:]

            # Select two random solutions from the fittest solutions
            parent1, parent2 = random.sample(fittest_indices, 2)

            # Create a new solution by crossover and mutation
            child = np.copy(population[parent1])
            child[np.random.randint(0, self.dim, size=len(parent1))] = parent2[np.random.randint(0, self.dim, size=len(parent2))]
            child = np.clip(child, -5.0, 5.0)

            # Evaluate the new solution
            new_fitness = fitness(child)

            # Accept the new solution with a probability less than 1
            if np.random.rand() < 0.1:
                population[fittest_indices].append(child)
                fitness_values[fittest_indices] = new_fitness
            else:
                population.pop(fittest_indices)

            # Update the population size
            population_size -= 1

            # Update the best solution found so far
            if fitness_values[fittest_indices] < fitness_values[-1]:
                population = [child]
                fitness_values = [new_fitness]
            else:
                population.append(child)
                fitness_values.append(new_fitness)

        # Return the optimal solution and the number of function evaluations used
        return population[0], self.budget - len(population)


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

An exception occured: Traceback (most recent call last):
  File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
    new_individual = self.evaluate_fitness(new_individual)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/LLaMEA/llamea/llamea.py", line 264, in evaluate_fitness
    updated_individual = self.f(individual, self.logger)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 52, in evaluateBBOB
NameError: name'self' is not defined
.

# Description: Novel Genetic Algorithm for Black Box Optimization using Dynamic Crossover and Mutation
# Code: 