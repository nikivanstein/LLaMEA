import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def novel_metaheuristic(self, func, population_size=100, mutation_rate=0.01, crossover_rate=0.5, elite_size=10, iterations=1000):
        """
        Novel metaheuristic algorithm for black box optimization.

        Args:
            func: The black box function to optimize.
            population_size (int): The size of the population. Default is 100.
            mutation_rate (float): The probability of mutation. Default is 0.01.
            crossover_rate (float): The probability of crossover. Default is 0.5.
            elite_size (int): The size of the elite population. Default is 10.
            iterations (int): The number of iterations. Default is 1000.

        Returns:
            A list of tuples, where each tuple contains an individual and its fitness.
        """
        # Initialize the population
        population = [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(population_size)]

        # Initialize the elite population
        elite = population[:elite_size]

        # Run the algorithm for the specified number of iterations
        for _ in range(iterations):
            # Select parents using crossover and mutation
            parents = []
            while len(parents) < population_size:
                parent1, parent2 = random.sample(elite, 2)
                if random.random() < crossover_rate:
                    # Perform crossover
                    child = (1 - mutation_rate) * parent1 + mutation_rate * parent2
                else:
                    # Perform mutation
                    child = parent1 + random.uniform(-1, 1)
                parents.append(child)

            # Evaluate the fitness of the parents
            fitnesses = [func(child) for child in parents]

            # Select the fittest individuals
            elite = [parent for _, parent in sorted(zip(fitnesses, parents), reverse=True)[:elite_size]]

            # Replace the elite population with the fittest individuals
            population = elite + [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(population_size - elite_size)]

        # Return the final elite population
        return elite

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 