import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Black Box Optimizer.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the optimization space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function using evolutionary algorithm.

        Parameters:
        func (callable): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size and the number of generations
        pop_size = 100
        num_generations = 100

        # Initialize the population with random parameters
        self.population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))

        # Run the evolutionary algorithm
        for gen in range(num_generations):
            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Select the fittest individuals based on their mean and standard deviation
            mean = np.mean(evaluations)
            std = np.std(evaluations)
            indices = np.argsort([mean - std * i for i in range(pop_size)])

            # Select the fittest individuals with a probability of 0.4
            selected_indices = np.random.choice(indices, pop_size, replace=False, p=[0.4, 0.6])

            # Mutate the selected individuals
            mutated_indices = selected_indices.copy()
            mutated_population = self.mutate(self.population, mutated_indices)

            # Evaluate the function at each individual in the mutated population
            mutated_evaluations = [func(mutated_individual) for mutated_individual in mutated_population]

            # Check if the population has reached the budget
            if len(mutated_evaluations) < self.budget:
                break

        # Return the optimized parameters and the optimized function value
        return self.population, mutated_evaluations[-1]

    def select_fittest(self, pop_size, evaluations):
        """
        Select the fittest individuals in the population.

        Parameters:
        pop_size (int): The size of the population.
        evaluations (list): The function values of the individuals in the population.

        Returns:
        np.ndarray: The indices of the fittest individuals.
        """
        # Calculate the mean and standard deviation of the function values
        mean = np.mean(evaluations)
        std = np.std(evaluations)

        # Select the fittest individuals based on their mean and standard deviation
        indices = np.argsort([mean - std * i for i in range(pop_size)])

        return indices

    def mutate(self, population, indices):
        """
        Mutate the selected individuals.

        Parameters:
        population (np.ndarray): The selected individuals.
        indices (np.ndarray): The indices of the selected individuals.

        Returns:
        np.ndarray: The mutated individuals.
        """
        # Create a copy of the population
        mutated = population.copy()

        # Randomly swap two individuals in the population
        for i in range(len(mutated)):
            j = np.random.choice(len(mutated))
            mutated[i], mutated[j] = mutated[j], mutated[i]

        return mutated

# Example usage
def objective_function(x):
    return np.sum(x**2)

optimizer = BlackBoxOptimizer(100, 10)
optimized_parameters, optimized_function_value = optimizer(__call__(objective_function))

# Print the optimized parameters and the optimized function value
print(f"Optimized parameters: {optimized_parameters}")
print(f"Optimized function value: {optimized_function_value}")