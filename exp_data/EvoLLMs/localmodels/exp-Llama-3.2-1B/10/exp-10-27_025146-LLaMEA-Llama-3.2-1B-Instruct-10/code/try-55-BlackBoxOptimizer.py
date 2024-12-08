# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

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

def mutation_exp(individual, budget, dim):
    """
    Apply mutation to the individual.

    Args:
        individual (List[float]): The individual to mutate.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

    Returns:
        List[float]: The mutated individual.
    """
    # Randomly select an index to mutate
    index = np.random.randint(0, dim)

    # Mutate the individual by adding a random value between -1 and 1
    mutated_individual = individual.copy()
    mutated_individual[index] += random.uniform(-1, 1)

    # Ensure the mutated individual stays within the search space
    mutated_individual[index] = max(-5.0, min(5.0, mutated_individual[index]))

    return mutated_individual

def selection_exp(individuals, budget, dim):
    """
    Select the best individual using tournament selection.

    Args:
        individuals (List[List[float]]): The list of individuals.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

    Returns:
        List[float]: The selected individual.
    """
    # Select the top individual using tournament selection
    selected_individual = individuals[np.random.choice(len(individuals), budget, replace=False)]

    return selected_individual

def crossover_exp(parent1, parent2, budget, dim):
    """
    Perform crossover between two parents.

    Args:
        parent1 (List[float]): The first parent.
        parent2 (List[float]): The second parent.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

    Returns:
        List[float]: The child individual.
    """
    # Randomly select a crossover point
    crossover_point = np.random.randint(0, dim)

    # Split the parents into two halves
    left_child = parent1[:crossover_point]
    right_child = parent1[crossover_point:]

    # Perform crossover
    child = left_child + [random.uniform(left_child[-1], right_child[-1])]

    return child

def selection_exp(individuals, budget, dim):
    """
    Select the best individual using roulette wheel selection.

    Args:
        individuals (List[List[float]]): The list of individuals.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

    Returns:
        List[float]: The selected individual.
    """
    # Calculate the probability of selection for each individual
    probabilities = np.array([1 / len(individuals) for _ in individuals])

    # Select the individual with the highest probability
    selected_individual = individuals[np.argmax(probabilities)]

    return selected_individual

def genetic_exp(individuals, budget, dim):
    """
    Perform genetic algorithm.

    Args:
        individuals (List[List[float]]): The list of individuals.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

    Returns:
        List[float]: The optimized individual.
    """
    # Initialize the population
    population = individuals

    # Perform the specified number of generations
    for _ in range(100):
        # Select the best individual using roulette wheel selection
        selected_individual = selection_exp(population, budget, dim)

        # Perform crossover between two parents
        child = crossover_exp(selected_individual, selected_individual, budget, dim)

        # Mutate the child
        child = mutation_exp(child, budget, dim)

        # Replace the least fit individual in the population
        population[np.argmin(population)] = child

    return population[0]

# Initialize the BlackBoxOptimizer
optimizer = BlackBoxOptimizer(1000, 5)

# Evaluate the function
def func(individual):
    """
    Evaluate the function at the given individual.

    Args:
        individual (List[float]): The individual to evaluate.

    Returns:
        float: The value of the function.
    """
    # Evaluate the function using the BlackBoxOptimizer
    return optimizer.func(individual)

# Optimize the function using the genetic algorithm
individual = genetic_exp([func(individual) for individual in range(100)], 1000, 5)

# Print the result
print("Optimized value:", func(individual))