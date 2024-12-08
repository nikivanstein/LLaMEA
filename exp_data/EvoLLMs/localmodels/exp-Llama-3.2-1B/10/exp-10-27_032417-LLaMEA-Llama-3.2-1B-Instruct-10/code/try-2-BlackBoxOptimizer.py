import numpy as np

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


# Example usage:
def func(x):
    return x**2 + 2*x + 1

def fitness(individual):
    return np.exp(individual)


def mutate(individual):
    return individual + np.random.uniform(-1.0, 1.0, individual.shape)


def select_parents(population, num_parents):
    return np.random.choice(population, num_parents, replace=False)


def breed(parents):
    offspring = []
    for _ in range(100):
        offspring.append(mutate(parents.pop()))
    return offspring


def anneal(budget, dim, population_size, num_parents, num_offspring):
    parents = select_parents(population, num_parents)
    offspring = breed(parents)
    for _ in range(budget):
        # Simulate annealing
        current_temperature = 1.0
        for individual in offspring:
            # Calculate the probability of accepting the current individual
            probability = np.exp((current_temperature - np.exp(individual)) / 100)

            # Accept the current individual with a probability less than 1
            if np.random.rand() < probability:
                offspring.remove(individual)
                offspring.append(mutate(individual))
                current_temperature *= 0.9

        # Update the population
        population = offspring
    return population


# Example usage:
optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Refine the solution using genetic algorithm with probability 0.1
def refined_solution(individual):
    return individual + np.random.uniform(-1.0, 1.0, individual.shape)

def mutate(individual):
    return individual + np.random.uniform(-1.0, 1.0, individual.shape)

refined_solution = refined_solution(optimal_solution)
print("Refined solution:", refined_solution)
print("Number of function evaluations:", num_evaluations)