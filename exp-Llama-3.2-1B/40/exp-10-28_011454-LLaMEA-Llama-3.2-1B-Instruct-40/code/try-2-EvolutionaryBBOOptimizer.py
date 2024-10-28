import numpy as np
from scipy.optimize import minimize

class EvolutionaryBBOOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Evolutionary BBO Optimizer.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the optimization space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func, mutation_prob=0.4):
        """
        Optimize the black box function using evolutionary algorithm with adaptive mutation strategy.

        Parameters:
        func (callable): The black box function to optimize.
        mutation_prob (float): The probability of mutation (default: 0.4).

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

            # Select the fittest individuals
            self.population = self.select_fittest(pop_size, evaluations)

            # Select the fittest individuals based on their mean and standard deviation
            indices = np.argsort([mean - std * i for i, mean in enumerate(evaluations) for std in [std * i for std in np.linspace(-5.0, 5.0, pop_size)]])

            # Select the fittest 50% of the population
            self.population = self.population[:pop_size // 2][indices]

            # Mutate the selected individuals
            for i in range(pop_size):
                if np.random.rand() < mutation_prob:
                    # Randomly swap two individuals in the population
                    j = np.random.choice(pop_size)
                    self.population[i], self.population[j] = self.population[j], self.population[i]

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

# Example usage
def sphere_func(x):
    return x[0]**2 + x[1]**2

optimizer = EvolutionaryBBOOptimizer(budget=1000, dim=2)
optimized_params, optimized_func_value = optimizer(sphere_func)

# Print the optimized parameters and function value
print("Optimized Parameters:", optimized_params)
print("Optimized Function Value:", optimized_func_value)