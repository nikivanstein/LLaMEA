import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_rate=0.1, crossover_rate=0.5, learning_rate=0.01):
        """
        Initialize the Adaptive Black Box Optimizer.

        Args:
        - budget (int): The number of function evaluations.
        - dim (int): The dimensionality of the search space.
        - mutation_rate (float, optional): The probability of mutation. Defaults to 0.1.
        - crossover_rate (float, optional): The probability of crossover. Defaults to 0.5.
        - learning_rate (float, optional): The learning rate for the evolutionary strategy. Defaults to 0.01.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate

    def __call__(self, func):
        """
        Evaluate the function with the given budget and select the top-performing individuals.

        Args:
        - func (callable): The black box function to optimize.

        Returns:
        - individual (float): The best individual in the current population.
        """
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])

        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]

        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)

        # Replace the old population with the new one
        self.population = new_population

        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])

        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def adaptive EvolutionaryStrategy(self, func):
        """
        Use the evolutionary strategy to refine the selected solution.

        Args:
        - func (callable): The black box function to optimize.

        Returns:
        - best_individual (float): The best individual in the current population.
        """
        # Initialize the population with the current best individual
        population = [self.__call__(func)]

        # Run the evolutionary strategy for the specified budget
        for _ in range(self.budget):
            # Select the top-performing individuals
            top_individuals = np.argsort(func(population[-1]))[-self.population_size:]

            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(top_individuals, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child = random.uniform(self.search_space[0], self.search_space[1])
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

        # Return the best individual
        best_individual = np.argmax(func(population[-1]))
        return best_individual