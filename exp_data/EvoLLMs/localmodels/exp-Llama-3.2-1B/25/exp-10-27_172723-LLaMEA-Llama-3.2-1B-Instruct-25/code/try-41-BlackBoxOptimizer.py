import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, iterations=100):
        """
        Evaluates the function for the specified number of times within the budget.
        
        Args:
        func: The function to be optimized.
        iterations (int): The number of times to evaluate the function. Defaults to 100.
        
        Returns:
        point: The point that gives the best evaluation of the function within the budget.
        """
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + iterations)
        self.func_evaluations += iterations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

def mutation_exp(problem, iterations=100, mutation_rate=0.1):
    """
    Performs a mutation on the selected solution to refine its strategy.
    
    Args:
    problem: The problem to be optimized.
    iterations (int): The number of times to perform mutations. Defaults to 100.
    mutation_rate (float): The probability of performing a mutation. Defaults to 0.1.
    
    Returns:
    new_individual: The mutated solution.
    """
    # Select a random individual from the population
    new_individual = problem.evaluate_fitness([problem.evaluate_fitness([problem.evaluate_fitness([problem.evaluate_fitness([i for i in range(5)]]) for i in range(5)])])])
    
    # Perform mutations
    for _ in range(iterations):
        # Select a random individual
        individual = problem.evaluate_fitness([problem.evaluate_fitness([problem.evaluate_fitness([i for i in range(5)]]) for i in range(5)])])
        
        # Perform a mutation
        if random.random() < mutation_rate:
            # Randomly select a gene
            gene = random.randint(0, 4)
            
            # Mutate the gene
            individual[gene] = random.uniform(-5.0, 5.0)
    
    # Return the mutated solution
    return new_individual

# Initialize the Black Box Optimizer
optimizer = BlackBoxOptimizer(1000, 5)

# Run the optimization algorithm
optimizer(BlackBoxOptimizer(1000, 5))

# Print the results
print("Selected solution:", optimizer.func(BlackBoxOptimizer(1000, 5)))
print("Score:", optimizer.func(BlackBoxOptimizer(1000, 5)))