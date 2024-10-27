# Description: Evolutionary Optimization Algorithm for Black Box Functions
# Code:
import numpy as np
import random
import time

class EvolutionaryOptimization:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the evolutionary optimization algorithm.

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
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize the population with random parameter values.

        Returns:
            list: A list of dictionaries containing the parameter values and their fitness values.
        """
        population = []
        for _ in range(self.population_size):
            individual = {}
            for i in range(self.dim):
                individual[f"param_{i}"] = np.random.uniform(-5.0, 5.0, 1)
            population.append(individual)
        return population

    def selection(self, population):
        """
        Select the fittest individuals for the next generation.

        Args:
            population (list): A list of dictionaries containing the parameter values and their fitness values.

        Returns:
            list: A list of dictionaries containing the selected individuals.
        """
        fitness_values = [individual["fitness"] for individual in population]
        sorted_indices = np.argsort(fitness_values)
        selected_indices = sorted_indices[:self.population_size // 2]
        selected_population = [population[i] for i in selected_indices]
        return selected_population

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create a child.

        Args:
            parent1 (dict): The first parent.
            parent2 (dict): The second parent.

        Returns:
            dict: The child.
        """
        child = {}
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                child[f"param_{i}"] = parent1[f"param_{i}"] + parent2[f"param_{i}"]
            else:
                child[f"param_{i}"] = parent2[f"param_{i}"]
        return child

    def mutation(self, individual):
        """
        Perform mutation on an individual to introduce noise.

        Args:
            individual (dict): The individual.

        Returns:
            dict: The mutated individual.
        """
        for i in range(self.dim):
            if np.random.rand() < self.noise_level:
                individual[f"param_{i}"] += np.random.normal(0, 1, 1)
        return individual

    def fitness(self, individual):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (dict): The individual.

        Returns:
            float: The fitness value.
        """
        func_value = individual["func"](*individual["param"])
        return func_value

    def __call__(self, func):
        """
        Optimize the black box function `func` using evolutionary optimization.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population
        population = self.initialize_population()
        for _ in range(self.budget):
            # Select the fittest individuals
            selected_population = self.selection(population)
            # Perform crossover and mutation
            children = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(selected_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                children.append(child)
            # Replace the population with the new generation
            population = selected_population + children
        # Return the optimized parameter values and the objective function value
        return self.evaluate_fitness(population)

# Description: Evolutionary Optimization Algorithm for Black Box Functions
# Code:
# ```python
# import numpy as np
# import random
# import time

# class EvolutionaryOptimization:
#     def __init__(self, budget, dim, noise_level=0.1):
#         """
#         Initialize the evolutionary optimization algorithm.

#         Args:
#             budget (int): The maximum number of function evaluations allowed.
#             dim (int): The dimensionality of the problem.
#             noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
#         """
#         self.budget = budget
#         self.dim = dim
#         self.noise_level = noise_level
#         self.noise = 0
#         self.population_size = 100
#         self.population = self.initialize_population()

#     def initialize_population(self):
#         """
#         Initialize the population with random parameter values.

#         Returns:
#             list: A list of dictionaries containing the parameter values and their fitness values.
#         """
#         population = []
#         for _ in range(self.population_size):
#             individual = {}
#             for i in range(self.dim):
#                 individual[f"param_{i}"] = np.random.uniform(-5.0, 5.0, 1)
#             population.append(individual)
#         return population

#     def selection(self, population):
#         """
#         Select the fittest individuals for the next generation.

#         Args:
#             population (list): A list of dictionaries containing the parameter values and their fitness values.

#         Returns:
#             list: A list of dictionaries containing the selected individuals.
#         """
#         fitness_values = [individual["fitness"] for individual in population]
#         sorted_indices = np.argsort(fitness_values)
#         selected_indices = sorted_indices[:self.population_size // 2]
#         selected_population = [population[i] for i in selected_indices]
#         return selected_population

#     def crossover(self, parent1, parent2):
#         """
#         Perform crossover between two parents to create a child.

#         Args:
#             parent1 (dict): The first parent.
#             parent2 (dict): The second parent.

#         Returns:
#             dict: The child.
#         """
#         child = {}
#         for i in range(self.dim):
#             if np.random.rand() < 0.5:
#                 child[f"param_{i}"] = parent1[f"param_{i}"] + parent2[f"param_{i}"]
#             else:
#                 child[f"param_{i}"] = parent2[f"param_{i}"]
#         return child

#     def mutation(self, individual):
#         """
#         Perform mutation on an individual to introduce noise.

#         Args:
#             individual (dict): The individual.

#         Returns:
#             dict: The mutated individual.
#         """
#         for i in range(self.dim):
#             if np.random.rand() < self.noise_level:
#                 individual[f"param_{i}"] += np.random.normal(0, 1, 1)
#         return individual

#     def fitness(self, individual):
#         """
#         Evaluate the fitness of an individual.

#         Args:
#             individual (dict): The individual.

#         Returns:
#             float: The fitness value.
#         """
#         func_value = individual["func"](*individual["param"])
#         return func_value

#     def __call__(self, func):
#         """
#         Optimize the black box function `func` using evolutionary optimization.

#         Args:
#             func (callable): The black box function to optimize.

#         Returns:
#             tuple: A tuple containing the optimized parameter values and the objective function value.
#         """
#         # Initialize the population
#         population = self.initialize_population()
#         for _ in range(self.budget):
#             # Select the fittest individuals
#             selected_population = self.selection(population)
#             # Perform crossover and mutation
#             children = []
#             for _ in range(self.population_size // 2):
#                 parent1, parent2 = random.sample(selected_population, 2)
#                 child = self.crossover(parent1, parent2)
#                 child = self.mutation(child)
#                 children.append(child)
#             # Replace the population with the new generation
#             population = selected_population + children
#         # Return the optimized parameter values and the objective function value
#         return self.evaluate_fitness(population)

# # Description: Evolutionary Optimization Algorithm for Black Box Functions
# # Code:
# # ```python
# # import numpy as np
# # import random
# # import time

# # class EvolutionaryOptimization:
# #     def __init__(self, budget, dim, noise_level=0.1):
# #         """
# #         Initialize the evolutionary optimization algorithm.

# #         Args:
# #             budget (int): The maximum number of function evaluations allowed.
# #             dim (int): The dimensionality of the problem.
# #             noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
# #         """
# #         self.budget = budget
# #         self.dim = dim
# #         self.noise_level = noise_level
# #         self.noise = 0
# #         self.population_size = 100
# #         self.population = self.initialize_population()

# #     def initialize_population(self):
# #         """
# #         Initialize the population with random parameter values.

# #         Returns:
# #             list: A list of dictionaries containing the parameter values and their fitness values.
# #         """
# #         population = []
# #         for _ in range(self.population_size):
# #             individual = {}
# #             for i in range(self.dim):
# #                 individual[f"param_{i}"] = np.random.uniform(-5.0, 5.0, 1)
# #             population.append(individual)
# #         return population

# #     def selection(self, population):
# #         """
# #         Select the fittest individuals for the next generation.

# #         Args:
# #             population (list): A list of dictionaries containing the parameter values and their fitness values.

# #         Returns:
# #             list: A list of dictionaries containing the selected individuals.
# #         """
# #         fitness_values = [individual["fitness"] for individual in population]
# #         sorted_indices = np.argsort(fitness_values)
# #         selected_indices = sorted_indices[:self.population_size // 2]
# #         selected_population = [population[i] for i in selected_indices]
# #         return selected_population

# #     def crossover(self, parent1, parent2):
# #         """
# #         Perform crossover between two parents to create a child.

# #         Args:
# #             parent1 (dict): The first parent.
# #             parent2 (dict): The second parent.

# #         Returns:
# #             dict: The child.
# #         """
# #         child = {}
# #         for i in range(self.dim):
# #             if np.random.rand() < 0.5:
# #                 child[f"param_{i}"] = parent1[f"param_{i}"] + parent2[f"param_{i}"]
# #             else:
# #                 child[f"param_{i}"] = parent2[f"param_{i}"]
# #         return child

# #     def mutation(self, individual):
# #         """
# #         Perform mutation on an individual to introduce noise.

# #         Args:
# #             individual (dict): The individual.

# #         Returns:
# #             dict: The mutated individual.
# #         """
# #         for i in range(self.dim):
# #             if np.random.rand() < self.noise_level:
# #                 individual[f"param_{i}"] += np.random.normal(0, 1, 1)
# #         return individual

# #     def fitness(self, individual):
# #         """
# #         Evaluate the fitness of an individual.

# #         Args:
# #             individual (dict): The individual.

# #         Returns:
# #             float: The fitness value.
# #         """
# #         func_value = individual["func"](*individual["param"])
# #         return func_value

# #     def __call__(self, func):
# #         """
# #         Optimize the black box function `func` using evolutionary optimization.

# #         Args:
# #             func (callable): The black box function to optimize.

# #         Returns:
# #             tuple: A tuple containing the optimized parameter values and the objective function value.
# #         """
# #         # Initialize the population
# #         population = self.initialize_population()
# #         for _ in range(self.budget):
# #             # Select the fittest individuals
# #             selected_population = self.selection(population)
# #             # Perform crossover and mutation
# #             children = []
# #             for _ in range(self.population_size // 2):
# #                 parent1, parent2 = random.sample(selected_population, 2)
# #                 child = self.crossover(parent1, parent2)
# #                 child = self.mutation(child)
# #                 children.append(child)
# #             # Replace the population with the new generation
# #             population = selected_population + children
# #         # Return the optimized parameter values and the objective function value
# #         return self.evaluate_fitness(population)