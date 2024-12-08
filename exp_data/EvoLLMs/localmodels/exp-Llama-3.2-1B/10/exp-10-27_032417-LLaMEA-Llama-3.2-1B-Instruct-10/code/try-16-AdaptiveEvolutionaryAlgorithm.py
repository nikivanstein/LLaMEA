import numpy as np

class AdaptiveEvolutionaryAlgorithm:
    """
    A novel metaheuristic algorithm for solving black box optimization problems.

    The algorithm combines evolutionary algorithms with adaptive probability of acceptance to find the optimal solution.
    """

    def __init__(self, budget, dim):
        """
        Initializes the algorithm with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.acceptance_rate = 0.1

    def __call__(self, func):
        """
        Optimizes a black box function using the algorithm.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the population with random solutions
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the black box function at each solution
        evaluations = 0
        while evaluations < self.budget:
            # Evaluate the black box function at each solution
            func(population[evaluations])

            # Generate a new solution by mutating the current population
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    population[i] = np.random.uniform(-5.0, 5.0, self.dim)

            # Select the fittest solutions to reproduce
            fittest_indices = np.argsort(np.linalg.norm(population, axis=1))[:self.population_size // 2]
            population[fittest_indices] = np.random.uniform(-5.0, 5.0, (self.population_size // 2, self.dim))

            # Evaluate the black box function at the fittest solutions
            evaluations += 1

            # Accept the fittest solutions with a probability less than the acceptance rate
            if evaluations > 0:
                probability = np.exp((evaluations - evaluations) / self.budget)
                if np.random.rand() < probability:
                    population = population[fittest_indices]

        # Return the optimal solution and the number of function evaluations used
        return population[np.argmax(np.linalg.norm(population, axis=1))], evaluations