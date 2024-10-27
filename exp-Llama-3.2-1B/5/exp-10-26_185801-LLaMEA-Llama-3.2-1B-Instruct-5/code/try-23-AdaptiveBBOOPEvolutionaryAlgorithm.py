import numpy as np

class AdaptiveBBOOPEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        """
        Initialize the algorithm with a given budget and dimensionality.
        
        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        """
        Optimize the black box function using evolutionary exploration.
        
        Args:
        func (function): The black box function to optimize.
        
        Returns:
        float: The optimized function value.
        """
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def __str__(self):
        return "Adaptive Black Box Optimization with Evolutionary Exploration"

    def __repr__(self):
        return f"AdaptiveBBOOPEvolutionaryAlgorithm(budget={self.budget}, dim={self.dim})"

    def adaptive_exploration(self, mutation_rate, exploration_rate):
        """
        Refine the solution by adjusting the mutation rate and exploration rate.
        
        Args:
        mutation_rate (float): The current mutation rate.
        exploration_rate (float): The current exploration rate.
        
        Returns:
        tuple: A tuple containing the new mutation rate and exploration rate.
        """
        if self.func_evaluations >= self.budget:
            # If the budget is reached, return the current solution
            return mutation_rate, exploration_rate
        else:
            # If the budget is not reached, refine the solution
            new_mutation_rate = self.mutation_rate * exploration_rate
            new_exploration_rate = self.exploration_rate * mutation_rate
            return new_mutation_rate, new_exploration_rate

    def run(self, func, mutation_rate, exploration_rate):
        """
        Run the algorithm with the specified mutation rate and exploration rate.
        
        Args:
        func (function): The black box function to optimize.
        mutation_rate (float): The mutation rate to use.
        exploration_rate (float): The exploration rate to use.
        
        Returns:
        float: The optimized function value.
        """
        new_individual = self.evaluate_fitness(func)
        mutation_rate, exploration_rate = self.adaptive_exploration(mutation_rate, exploration_rate)
        return self.__call__(func, new_individual)

# Example usage:
def example_func(x):
    return np.sin(x)

algorithm = AdaptiveBBOOPEvolutionaryAlgorithm(budget=100, dim=5)
optimized_func = algorithm.run(example_func, mutation_rate=0.1, exploration_rate=0.5)
print(f"Optimized function value: {optimized_func}")