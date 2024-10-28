import numpy as np
import random

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = None
        self.fitness_scores = None
        self.population_size = 100
        self.evolved_individuals = []
        self.evolved_fitness = []

    def __call__(self, func):
        """
        Optimize the black box function using Adaptive Genetic Algorithm.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize population with random individuals
        self.population = np.random.rand(self.population_size, self.dim)
        # Initialize fitness scores
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        # Run the optimization algorithm
        for _ in range(self.budget):
            # Generate a random fitness score
            fitness_scores = self.evaluate_fitness(self.population)
            # Select the fittest individuals
            self.evolved_individuals = self.select_fittest_individuals(fitness_scores)
            # Evaluate the fitness of the fittest individuals
            self.fitness_scores = fitness_scores
            # Update the population
            self.population = np.vstack((self.population, self.evolved_individuals))
            # Check if the optimization is successful
            if np.allclose(self.population, self.fitness_scores):
                break
        # Return the fittest individual
        return self.population[np.argmax(self.fitness_scores)]

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (numpy array): The individual to evaluate.

        Returns:
            float: The fitness score of the individual.
        """
        # Define the neural network architecture
        self.nn = {
            'input': self.dim,
            'hidden': self.dim,
            'output': 1
        }
        # Define the optimization function
        def optimize(x):
            # Forward pass
            y = np.dot(x, self.nn['input'].reshape(-1, 1)) + self.nn['output'].reshape(1, 1)
            # Backward pass
            dy = np.dot(self.nn['output'].reshape(-1, 1), (y - func(x)))
            # Update weights and bias
            self.nn['input'].reshape(-1, 1) -= 0.1 * dy * x
            self.nn['output'].reshape(1, 1) -= 0.1 * dy
            return y
        # Run the optimization algorithm
        for _ in range(100):
            # Generate a random input
            x = np.random.rand(self.dim)
            # Optimize the function
            y = optimize(x)
            # Check if the optimization is successful
            if np.allclose(y, func(x)):
                return y
        # If the optimization fails, return None
        return None

    def select_fittest_individuals(self, fitness_scores):
        """
        Select the fittest individuals based on their fitness scores.

        Args:
            fitness_scores (numpy array): The fitness scores of the individuals.

        Returns:
            numpy array: The fittest individuals.
        """
        # Sort the individuals based on their fitness scores
        self.evolved_individuals = np.sort(fitness_scores, axis=1)
        # Return the fittest individuals
        return self.evolved_individuals[:, :self.population_size // 2]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = AdaptiveGeneticAlgorithm(1000, 2)
optimized_value = optimizer(func)
print(optimized_value)