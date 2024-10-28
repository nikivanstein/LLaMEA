import numpy as np
import random
import math

class NeuralOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.weights = None
        self.bias = None
        self.mutation_rate = 0.1

    def __call__(self, func):
        """
        Optimize the black box function using Neural Optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize weights and bias using a neural network
        self.weights = np.random.rand(self.dim)
        self.bias = np.random.rand(1)
        self.weights = np.vstack((self.weights, [0]))
        self.bias = np.append(self.bias, 0)

        # Define the neural network architecture
        self.nn = {
            'input': self.dim,
            'hidden': self.dim,
            'output': 1
        }

        # Define the optimization function
        def optimize(x):
            # Forward pass
            y = np.dot(x, self.weights) + self.bias
            # Backward pass
            dy = np.dot(self.nn['output'].reshape(-1, 1), (y - func(x)))
            # Update weights and bias
            self.weights -= 0.1 * dy * x
            self.bias -= 0.1 * dy
            return y

        # Run the optimization algorithm
        for _ in range(self.budget):
            # Generate a random input
            x = np.random.rand(self.dim)
            # Optimize the function
            y = optimize(x)
            # Check if the optimization is successful
            if np.allclose(y, func(x)):
                return y

        # If the optimization fails, return None
        return None

class AdaptiveMutationNeuralOptimizer(NeuralOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_count = 0

    def mutate(self, x):
        """
        Mutate the individual using a combination of mutation and adaptive mutation strategy.

        Args:
            x (numpy array): The individual to mutate.

        Returns:
            numpy array: The mutated individual.
        """
        # Calculate the mutation probability
        mutation_prob = random.random()

        # If the mutation probability is less than the adaptive mutation rate, use the adaptive mutation strategy
        if mutation_prob < self.mutation_rate:
            # Generate a random mutation vector
            mutation_vector = np.random.rand(self.dim)
            # Apply the adaptive mutation strategy
            self.mutation_vector = mutation_vector * 0.5 + self.mutation_vector
            # Normalize the mutation vector
            self.mutation_vector = self.mutation_vector / np.linalg.norm(self.mutation_vector)
            # Apply the mutation to the individual
            x += self.mutation_vector * 0.1
        # Return the mutated individual
        return x

    def evaluate_fitness(self, func, individual):
        """
        Evaluate the fitness of the individual using the given function.

        Args:
            func (function): The function to evaluate the fitness.
            individual (numpy array): The individual to evaluate the fitness.

        Returns:
            float: The fitness of the individual.
        """
        # Apply the optimization algorithm
        optimized_individual = self.__call__(func, individual)
        # Evaluate the fitness of the optimized individual
        fitness = func(optimized_individual)
        # Return the fitness
        return fitness

# Example usage
def func(x):
    return x**2

optimizer = AdaptiveMutationNeuralOptimizer(budget=100, dim=2)
individual = np.random.rand(2)
fitness = optimizer.evaluate_fitness(func, individual)
print(fitness)