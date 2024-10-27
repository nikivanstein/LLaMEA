# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import differential_evolution

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

def novelty_metaheuristic(func, budget, dim):
    # Define the mutation strategy
    def mutation(individual):
        # Generate a new individual by changing one random element
        new_individual = individual.copy()
        new_individual[0] = random.uniform(-5.0, 5.0)
        return new_individual

    # Define the selection strategy
    def selection(population):
        # Use the tournament selection method
        tournament_size = 5
        tournament_results = []
        for _ in range(tournament_size):
            tournament_result = np.random.choice(population, size=dim, replace=False)
            tournament_results.append(func(tournament_result))
        return np.array(sorted(tournament_results))

    # Define the crossover strategy
    def crossover(parent1, parent2):
        # Use the uniform crossover method
        crossover_point = np.random.randint(0, dim)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    # Initialize the population
    population = [random.uniform(-5.0, 5.0) for _ in range(100)]

    # Perform the optimization
    for _ in range(budget):
        # Select the best individual
        best_individual = selection(population)

        # Evaluate the function at the best individual
        best_value = func(best_individual)

        # Crossover the best individual with another individual
        crossover_point1 = np.random.randint(0, dim)
        crossover_point2 = np.random.randint(0, dim)
        child1, child2 = crossover(best_individual, population[np.random.randint(0, 100)])

        # Mutation the child
        child1 = mutation(child1)
        child2 = mutation(child2)

        # Evaluate the function at the mutated child
        mutated_value = func(child1)
        mutated_value = func(child2)

        # Update the population
        population.append((best_value, best_individual))
        population.append((mutated_value, child1))
        population.append((mutated_value, child2))

        # Update the best individual
        if best_value > best_individual[0]:
            best_individual = (best_value, best_individual[1])

    # Return the best individual
    return best_individual[1]

# Test the algorithm
func = lambda x: x[0]**2 + x[1]**2
budget = 1000
dim = 2

best_individual = novelty_metaheuristic(func, budget, dim)
print("Optimized value:", best_individual[0])
print("Optimized individual:", best_individual[1])