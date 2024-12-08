# import numpy as np
# import random
# import operator

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
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_scores = self.initialize_fitness_scores()

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

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)

# One-line description with the main idea
# BlackBoxOptimizer: Genetic Algorithm for Black Box Optimization
# Code: 
# ```python
# import numpy as np
# import random
# import operator
# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population_size = 100
#         self.population = self.initialize_population()
#         self.fitness_scores = self.initialize_fitness_scores()

#     def __call__(self, func):
#         solution = None
#         evaluations = 0
#         while evaluations < self.budget:
#             solution = np.random.uniform(-5.0, 5.0, self.dim)
#             evaluations += 1
#             func(solution)
#             if evaluations > 0:
#                 probability = np.exp((evaluations - evaluations) / self.budget)
#                 if np.random.rand() < probability:
#                     solution = solution
#         return solution, evaluations

#     def initialize_population(self):
#         population = []
#         for _ in range(self.population_size):
#             individual = np.random.uniform(-5.0, 5.0, self.dim)
#             population.append(individual)
#         return population

#     def initialize_fitness_scores(self):
#         fitness_scores = []
#         for individual in self.population:
#             fitness_score = np.sum(np.square(individual))
#             fitness_scores.append(fitness_score)
#         return fitness_scores