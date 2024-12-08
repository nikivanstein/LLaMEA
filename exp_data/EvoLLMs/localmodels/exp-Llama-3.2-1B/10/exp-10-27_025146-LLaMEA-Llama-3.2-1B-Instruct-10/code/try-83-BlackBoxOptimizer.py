# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
import copy

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

def mutation_exp(individual, problem, mutation_rate):
    """
    Perform mutation on the individual.

    Args:
        individual (RealSingleObjectiveProblem): The individual to mutate.
        problem (RealSingleObjectiveProblem): The problem to optimize.
        mutation_rate (float): The probability of mutation.

    Returns:
        RealSingleObjectiveProblem: The mutated individual.
    """
    # Create a copy of the individual
    mutated_individual = copy.deepcopy(individual)

    # Randomly select a mutation point
    mutation_point = np.random.randint(0, mutated_individual.search_space.shape[0])

    # Perform the mutation
    mutated_individual.search_space[mutation_point] += random.uniform(-1, 1)

    # If the mutation rate is greater than 0, apply the mutation
    if random.random() < mutation_rate:
        mutated_individual.search_space[mutation_point] *= random.uniform(0.5, 2)

    return mutated_individual

def selection_exp(individuals, problem, num_individuals):
    """
    Select the best individuals.

    Args:
        individuals (list): The list of individuals.
        problem (RealSingleObjectiveProblem): The problem to optimize.
        num_individuals (int): The number of individuals to select.

    Returns:
        list: The selected individuals.
    """
    # Select the top num_individuals individuals
    selected_individuals = sorted(individuals, key=problem.f, reverse=True)[:num_individuals]

    return selected_individuals

def crossover_exp(parent1, parent2, problem, num_children):
    """
    Perform crossover on the parents.

    Args:
        parent1 (RealSingleObjectiveProblem): The first parent.
        parent2 (RealSingleObjectiveProblem): The second parent.
        problem (RealSingleObjectiveProblem): The problem to optimize.
        num_children (int): The number of children.

    Returns:
        list: The children.
    """
    # Randomly select the crossover points
    crossover_points = np.random.randint(0, parent1.search_space.shape[0], num_children)

    # Perform the crossover
    children = []
    for i in range(num_children):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1.search_space = np.concatenate((child1.search_space[:crossover_points[i]], parent2.search_space[crossover_points[i]:]))
        child2.search_space = np.concatenate((parent1.search_space[:crossover_points[i]], parent2.search_space[crossover_points[i]:]))
        children.append(crossover_exp(child1, child2, problem, 2))

    return children

def selection_exp(population, problem, num_individuals):
    """
    Select the best individuals.

    Args:
        population (list): The list of individuals.
        problem (RealSingleObjectiveProblem): The problem to optimize.
        num_individuals (int): The number of individuals to select.

    Returns:
        list: The selected individuals.
    """
    # Select the top num_individuals individuals
    selected_individuals = sorted(population, key=problem.f, reverse=True)[:num_individuals]

    return selected_individuals

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

def novel_metaheuristic_optimizer(budget, dim):
    """
    Novel Metaheuristic Algorithm for Black Box Optimization.

    Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

    Returns:
        BlackBoxOptimizer: The BlackBoxOptimizer instance.
    """
    # Initialize the BlackBoxOptimizer
    optimizer = BlackBoxOptimizer(budget, dim)

    # Evaluate a noiseless function
    func = lambda x: x**2
    optimized_value = optimizer(func)

    # Return the BlackBoxOptimizer instance
    return optimizer

# Example usage
budget = 100
dim = 10
optimizer = novel_metaheuristic_optimizer(budget, dim)

# Evaluate the function 5 times
for _ in range(5):
    optimized_value = optimizer.func()
    print(f"Optimized value: {optimized_value}")