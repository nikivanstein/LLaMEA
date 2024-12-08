import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100):
        """
        Optimizes the black box function `func` using `self.budget` function evaluations.
        
        Args:
            func (function): The black box function to optimize.
            budget (int, optional): The number of function evaluations. Defaults to 100.
        
        Returns:
            tuple: A tuple containing the optimized individual and its fitness.
        """
        population = self.generate_population(budget)
        while len(population) > 0:
            for _ in range(budget):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x, func(x)
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            population = self.update_population(population, func)
        return None, None

    def generate_population(self, budget):
        """
        Generates a population of individuals using a random search.
        
        Args:
            budget (int): The number of function evaluations.
        
        Returns:
            list: A list of individuals.
        """
        population = []
        for _ in range(budget):
            individual = np.random.uniform(self.search_space[:, 0], self.search_space[:, 1], size=self.dim)
            population.append(individual)
        return population

    def update_population(self, population, func):
        """
        Updates the population using a mutation strategy.
        
        Args:
            population (list): The current population.
            func (function): The black box function.
        
        Returns:
            list: The updated population.
        """
        updated_population = []
        for individual in population:
            new_individual = individual.copy()
            if np.random.rand() < 0.05:
                new_individual = new_individual + np.random.uniform(-5.0, 5.0, size=self.dim)
            else:
                new_individual = new_individual - np.random.uniform(-5.0, 5.0, size=self.dim)
            new_individual = self.evaluate_fitness(new_individual, func)
            updated_population.append(new_individual)
        return updated_population

    def evaluate_fitness(self, individual, func):
        """
        Evaluates the fitness of an individual using the black box function.
        
        Args:
            individual (numpy array): The individual.
            func (function): The black box function.
        
        Returns:
            float: The fitness of the individual.
        """
        return func(individual)

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# This algorithm uses a population-based approach to optimize black box functions by generating a population of individuals using a random search and updating the population using a mutation strategy.
# 
# The mutation strategy involves randomly changing the individual's coordinates with a small probability, which allows the algorithm to explore the search space and find better solutions.
# 
# The algorithm is designed to handle a wide range of tasks and can be easily adapted to different optimization problems by changing the mutation strategy.