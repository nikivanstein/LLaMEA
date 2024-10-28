import numpy as np
import random

class AdaptiveProbabilisticNeighborhoodSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.weights = None
        self.bias = None
        self.neighborhood = None
        self.fitness = None

    def __call__(self, func, population_size=100, mutation_rate=0.01):
        """
        Optimize the black box function using Adaptive Probabilistic Neighborhood Search.

        Args:
            func (function): The black box function to optimize.
            population_size (int): The size of the population. Defaults to 100.
            mutation_rate (float): The probability of mutation. Defaults to 0.01.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the population with random solutions
        self.population = self.generate_initial_population(population_size)
        self.fitness = np.zeros(len(self.population))
        for i in range(len(self.population)):
            self.fitness[i] = func(self.population[i])

        # Define the fitness function
        self.f = lambda x: -func(x)

        # Run the neighborhood search algorithm
        for _ in range(self.budget):
            # Generate a random solution
            new_individual = self.generate_new_individual()
            # Evaluate the fitness of the new individual
            new_fitness = self.f(new_individual)
            # Check if the new individual is better than the current best solution
            if new_fitness < self.fitness[i]:
                # Update the best solution
                self.population[i] = new_individual
                self.fitness[i] = new_fitness

            # Generate a new neighborhood
            neighborhood = [self.generate_new_individual() for _ in range(population_size)]
            # Evaluate the fitness of the neighborhood
            neighborhood_fitness = np.array([self.f(individual) for individual in neighborhood])
            # Calculate the probability of mutation
            mutation_probability = np.mean(neighborhood_fitness!= self.fitness)
            # Randomly select a new individual from the neighborhood
            if random.random() < mutation_probability:
                new_individual = random.choice(neighborhood)
            else:
                new_individual = self.generate_new_individual()

            # Update the population
            self.population = np.vstack((self.population, new_individual))

        # Return the best solution
        return self.population[np.argmax(self.fitness)]

    def generate_initial_population(self, population_size):
        """
        Generate the initial population with random solutions.

        Args:
            population_size (int): The size of the population.

        Returns:
            numpy.ndarray: The initial population.
        """
        return np.random.uniform(-5.0, 5.0, (population_size, self.dim))

    def generate_new_individual(self):
        """
        Generate a new individual by adding a small random perturbation to a solution.

        Returns:
            numpy.ndarray: The new individual.
        """
        return np.random.uniform(-5.0, 5.0, self.dim) + np.random.uniform(-0.1, 0.1, self.dim)

# Example usage
if __name__ == "__main__":
    # Define the black box function
    def func(x):
        return np.sin(x)

    # Initialize the Adaptive Probabilistic Neighborhood Search algorithm
    optimizer = AdaptiveProbabilisticNeighborhoodSearch(budget=100, dim=2)

    # Run the algorithm
    best_solution = optimizer(__call__, population_size=1000)
    print("Best solution:", best_solution)