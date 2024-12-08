# Description: Black Box Optimization using Evolutionary Algorithm with Adaptive Mutation
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Black Box Optimizer.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the optimization space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function using evolutionary algorithm.

        Parameters:
        func (callable): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size and the number of generations
        pop_size = 100
        num_generations = 100

        # Initialize the population with random parameters
        self.population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))

        # Run the evolutionary algorithm
        for gen in range(num_generations):
            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Select the fittest individuals
            self.population = self.select_fittest(pop_size, evaluations)

            # Adapt the mutation strategy based on the fitness values
            self.population = self.adapt_mutation_strategy(self.population, evaluations)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < self.budget:
                break

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

    def select_fittest(self, pop_size, evaluations):
        """
        Select the fittest individuals in the population.

        Parameters:
        pop_size (int): The size of the population.
        evaluations (list): The function values of the individuals in the population.

        Returns:
        np.ndarray: The indices of the fittest individuals.
        """
        # Calculate the mean and standard deviation of the function values
        mean = np.mean(evaluations)
        std = np.std(evaluations)

        # Select the fittest individuals based on their mean and standard deviation
        indices = np.argsort([mean - std * i for i in range(pop_size)])

        return indices

    def adapt_mutation_strategy(self, population, evaluations):
        """
        Adapt the mutation strategy based on the fitness values.

        Parameters:
        population (np.ndarray): The selected individuals.
        evaluations (list): The function values of the individuals in the population.

        Returns:
        np.ndarray: The mutated individuals.
        """
        # Initialize the mutation rate
        mutation_rate = 0.1

        # Create a copy of the population
        mutated = population.copy()

        # Randomly swap two individuals in the population with a probability based on their fitness values
        for i in range(len(mutated)):
            for j in range(i + 1, len(mutated)):
                if evaluations[i] > evaluations[j]:
                    k = np.random.choice([i, j])
                    mutated[k], mutated[j] = mutated[j], mutated[k]

        # Introduce adaptive mutation by changing the mutation rate based on the fitness values
        mutation_rate = 0.2 * evaluations[i] / evaluations[i] + 0.8 * evaluations[j] / evaluations[j]

        return mutated

# Description: Black Box Optimization using Evolutionary Algorithm with Adaptive Mutation
# Code: 
# ```python
# import numpy as np
# import random

# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         """
#         Initialize the Black Box Optimizer.

#         Parameters:
#         budget (int): The maximum number of function evaluations allowed.
#         dim (int): The dimensionality of the optimization space.
#         """
#         self.budget = budget
#         self.dim = dim

#     def __call__(self, func):
#         """
#         Optimize the black box function using evolutionary algorithm.

#         Parameters:
#         func (callable): The black box function to optimize.

#         Returns:
#         tuple: The optimized parameters and the optimized function value.
#         """
#         # Initialize the population size and the number of generations
#         pop_size = 100
#         num_generations = 100

#         # Initialize the population with random parameters
#         self.population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))

#         # Run the evolutionary algorithm
#         for gen in range(num_generations):
#             # Evaluate the function at each individual in the population
#             evaluations = [func(self.population[i]) for i in range(pop_size)]

#             # Select the fittest individuals
#             self.population = self.select_fittest(pop_size, evaluations)

#             # Adapt the mutation strategy based on the fitness values
#             self.population = self.adapt_mutation_strategy(self.population, evaluations)

#             # Evaluate the function at each individual in the population
#             evaluations = [func(self.population[i]) for i in range(pop_size)]

#             # Check if the population has reached the budget
#             if len(evaluations) < self.budget:
#                 break

#         # Return the optimized parameters and the optimized function value
#         return self.population, evaluations[-1]

#     def select_fittest(self, pop_size, evaluations):
#         """
#         Select the fittest individuals in the population.

#         Parameters:
#         pop_size (int): The size of the population.
#         evaluations (list): The function values of the individuals in the population.

#         Returns:
#         np.ndarray: The indices of the fittest individuals.
#         """
#         # Calculate the mean and standard deviation of the function values
#         mean = np.mean(evaluations)
#         std = np.std(evaluations)

#         # Select the fittest individuals based on their mean and standard deviation
#         indices = np.argsort([mean - std * i for i in range(pop_size)])

#         return indices

#     def adapt_mutation_strategy(self, population, evaluations):
#         """
#         Adapt the mutation strategy based on the fitness values.

#         Parameters:
#         population (np.ndarray): The selected individuals.
#         evaluations (list): The function values of the individuals in the population.

#         Returns:
#         np.ndarray: The mutated individuals.
#         """
#         # Initialize the mutation rate
#         mutation_rate = 0.1

#         # Create a copy of the population
#         mutated = population.copy()

#         # Randomly swap two individuals in the population with a probability based on their fitness values
#         for i in range(len(mutated)):
#             for j in range(i + 1, len(mutated)):
#                 if evaluations[i] > evaluations[j]:
#                     k = np.random.choice([i, j])
#                     mutated[k], mutated[j] = mutated[j], mutated[k]

#         # Introduce adaptive mutation by changing the mutation rate based on the fitness values
#         mutation_rate = 0.2 * evaluations[i] / evaluations[i] + 0.8 * evaluations[j] / evaluations[j]

#         return mutated

# # Description: Black Box Optimization using Evolutionary Algorithm with Adaptive Mutation
# # Code: 
# # ```python
# # import numpy as np
# # import random

# # class BlackBoxOptimizer:
# #     def __init__(self, budget, dim):
# #         """
# #         Initialize the Black Box Optimizer.

# #         Parameters:
# #         budget (int): The maximum number of function evaluations allowed.
# #         dim (int): The dimensionality of the optimization space.
# #         """
# #         self.budget = budget
# #         self.dim = dim

# #     def __call__(self, func):
# #         """
# #         Optimize the black box function using evolutionary algorithm.

# #         Parameters:
# #         func (callable): The black box function to optimize.

# #         Returns:
# #         tuple: The optimized parameters and the optimized function value.
# #         """
# #         # Initialize the population size and the number of generations
# #         pop_size = 100
# #         num_generations = 100

# #         # Initialize the population with random parameters
# #         self.population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))

# #         # Run the evolutionary algorithm
# #         for gen in range(num_generations):
# #             # Evaluate the function at each individual in the population
# #             evaluations = [func(self.population[i]) for i in range(pop_size)]

# #             # Select the fittest individuals
# #             self.population = self.select_fittest(pop_size, evaluations)

# #             # Adapt the mutation strategy based on the fitness values
# #             self.population = self.adapt_mutation_strategy(self.population, evaluations)

# #             # Evaluate the function at each individual in the population
# #             evaluations = [func(self.population[i]) for i in range(pop_size)]

# #             # Check if the population has reached the budget
# #             if len(evaluations) < self.budget:
# #                 break

# #         # Return the optimized parameters and the optimized function value
# #         return self.population, evaluations[-1]

# #     def select_fittest(self, pop_size, evaluations):
# #         """
# #         Select the fittest individuals in the population.

# #         Parameters:
# #         pop_size (int): The size of the population.
# #         evaluations (list): The function values of the individuals in the population.

# #         Returns:
# #         np.ndarray: The indices of the fittest individuals.
# #         """
# #         # Calculate the mean and standard deviation of the function values
# #         mean = np.mean(evaluations)
# #         std = np.std(evaluations)

# #         # Select the fittest individuals based on their mean and standard deviation
# #         indices = np.argsort([mean - std * i for i in range(pop_size)])

# #         return indices

# #     def adapt_mutation_strategy(self, population, evaluations):
# #         """
# #         Adapt the mutation strategy based on the fitness values.

# #         Parameters:
# #         population (np.ndarray): The selected individuals.
# #         evaluations (list): The function values of the individuals in the population.

# #         Returns:
# #         np.ndarray: The mutated individuals.
# #         """
# #         # Initialize the mutation rate
# #         mutation_rate = 0.1

# #         # Create a copy of the population
# #         mutated = population.copy()

# #         # Randomly swap two individuals in the population with a probability based on their fitness values
# #         for i in range(len(mutated)):
# #             for j in range(i + 1, len(mutated)):
# #                 if evaluations[i] > evaluations[j]:
# #                     k = np.random.choice([i, j])
# #                     mutated[k], mutated[j] = mutated[j], mutated[k]

# #         # Introduce adaptive mutation by changing the mutation rate based on the fitness values
# #         mutation_rate = 0.2 * evaluations[i] / evaluations[i] + 0.8 * evaluations[j] / evaluations[j]

# #         return mutated