import numpy as np

class AdaptiveGeneticAlgorithm:
    """
    A novel metaheuristic algorithm for solving black box optimization problems.
    """

    def __init__(self, budget, dim):
        """
        Initializes the optimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population_size_mutations = 10
        self.population_mutations = 10
        self.mutation_rate = 0.01
        self.adaptive_probability = 0.1

    def __call__(self, func):
        """
        Optimizes a black box function using the optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the solution and the number of function evaluations
        solution = None
        evaluations = 0

        # Initialize the population with random solutions
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

        # Evaluate the black box function for the initial population
        for _ in range(self.budget):
            evaluations += 1
            func(self.population[_])

            # Calculate the probability of accepting the current solution
            probability = np.exp((evaluations - evaluations) / self.budget)

            # Accept the current solution with a probability less than 1
            if np.random.rand() < probability:
                solution = self.population[_]
                break

        # Evolve the population using genetic algorithm
        while len(solution) < self.budget:
            # Select parents using tournament selection
            parents = np.array([self.select_parents(self.population, self.population_size) for _ in range(self.population_size)])

            # Create offspring by crossover and mutation
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = parents[np.random.randint(0, self.population_size, 2)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, self.population_mutations)
                offspring.append(child)

            # Replace the least fit individuals with the new offspring
            self.population = offspring

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


def func(x):
    return x**2 + 2*x + 1


# Example usage:
optimizer = AdaptiveGeneticAlgorithm(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# One-line description with the main idea
# Genetic Algorithm with Adaptive Probability of Acceptance
# Uses adaptive probability of acceptance to refine the solution strategy
# Evolves the population using genetic algorithm and tournament selection