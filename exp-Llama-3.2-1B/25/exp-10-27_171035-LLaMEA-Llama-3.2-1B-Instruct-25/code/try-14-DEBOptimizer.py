import numpy as np
from scipy.optimize import differential_evolution

class DEBOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the DEBOptimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None

    def __call__(self, func):
        """
        Optimize a black box function using DEBOptimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized function and its value.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize the population size and the number of generations
        population_size = 100
        num_generations = 100

        # Initialize the population with random solutions
        self.population = [np.random.uniform(lower_bound, upper_bound, size=(population_size, self.dim)) for _ in range(population_size)]

        # Evaluate the objective function for each individual in the population
        results = []
        for _ in range(num_generations):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Define the adaptive mutation strategy
        def adaptiveMutation(individual):
            # Get the fitness value of the individual
            fitness = -func(individual)

            # Get the best individual with the highest fitness value
            bestIndividual = self.population[np.argmax([fitness for _, fitness in results])]

            # Calculate the mutation probability
            mutationProbability = 0.25

            # Generate a new mutation candidate
            newIndividual = individual + np.random.normal(0, 1, self.dim)

            # Check if the mutation candidate is within the bounds
            if np.any(newIndividual < lower_bound) or np.any(newIndividual > upper_bound):
                # Apply the mutation
                newIndividual = bestIndividual + np.random.normal(0, 1, self.dim)

            # Apply the adaptive mutation strategy
            if np.random.rand() < mutationProbability:
                # Apply the mutation
                newIndividual = bestIndividual + np.random.normal(0, 1, self.dim)

            return newIndividual

        # Apply the adaptive mutation strategy to the population
        self.population = [adaptiveMutation(individual) for individual in self.population]

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])