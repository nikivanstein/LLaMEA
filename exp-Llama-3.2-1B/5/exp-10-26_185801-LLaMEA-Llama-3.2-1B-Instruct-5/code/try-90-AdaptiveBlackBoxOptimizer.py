import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_rate):
        """
        Initialize the AdaptiveBlackBoxOptimizer with a given budget, dimension, and mutation rate.

        Args:
        - budget (int): The maximum number of function evaluations allowed.
        - dim (int): The dimensionality of the search space.
        - mutation_rate (float): The probability of applying a mutation to an individual.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = mutation_rate
        self.new_individuals = []

    def __call__(self, func):
        """
        Optimize the given black box function using the AdaptiveBlackBoxOptimizer.

        Args:
        - func (function): The black box function to optimize.

        Returns:
        - func_value (float): The optimized function value.
        """
        while self.func_evaluations < self.budget:
            # Generate a new individual using the current search space
            new_individual = self.generate_new_individual()

            # Evaluate the new individual using the given function
            func_value = func(new_individual)

            # Check if the new individual is valid (no NaNs or infins)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")

            # Check if the new individual is within the valid range (0 to 1)
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")

            # Update the search space if the new individual is better
            if func_value > self.search_space[0]:
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            elif func_value < self.search_space[-1]:
                self.search_space = np.linspace(self.search_space[-1], 5.0, self.dim)

            # Apply mutation to the new individual with a probability of mutation_rate
            if np.random.rand() < self.mutation_rate:
                new_individual = self.mutate(new_individual)

            # Add the new individual to the list of new individuals
            self.new_individuals.append(new_individual)

            # Increment the function evaluation count
            self.func_evaluations += 1

        # Return the optimized function value
        return func_value

    def generate_new_individual(self):
        """
        Generate a new individual using the current search space.

        Returns:
        - new_individual (numpy array): The new individual.
        """
        return np.random.choice(self.search_space, size=self.dim, replace=False)

    def mutate(self, individual):
        """
        Mutate the given individual with a probability of mutation_rate.

        Args:
        - individual (numpy array): The individual to mutate.

        Returns:
        - mutated_individual (numpy array): The mutated individual.
        """
        mutated_individual = individual.copy()
        np.random.shuffle(mutated_individual)
        return mutated_individual