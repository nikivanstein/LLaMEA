import numpy as np
from scipy.optimize import differential_evolution

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

def evolutionary_algorithm(func, bounds, budget, dim):
    # Initialize the population
    population = [np.random.uniform(bounds, size=dim) for _ in range(100)]
    
    # Run the evolutionary algorithm for a specified number of generations
    for _ in range(100):
        # Evaluate the fitness of each individual in the population
        fitness = [func(individual) for individual in population]
        
        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[:int(budget/10)]
        
        # Mutate the fittest individuals
        for i in range(len(fittest_individuals)):
            if np.random.rand() < 0.25:
                # Refine the strategy by changing the lines of the selected individual
                fittest_individuals[i] = np.random.uniform(bounds, size=dim)
        
        # Replace the least fit individuals with the new fittest individuals
        population[fittest_individuals] = population[:fittest_individuals]
    
    # Return the fittest individual in the final population
    return population[np.argmax(fitness)]

# Define the test function
def test_func(x):
    return x[0]**2 + x[1]**2

# Run the evolutionary algorithm
best_individual = evolutionary_algorithm(test_func, [-5, -5], 1000, 2)

# Print the result
print("Best individual:", best_individual)
print("Fitness:", test_func(best_individual))