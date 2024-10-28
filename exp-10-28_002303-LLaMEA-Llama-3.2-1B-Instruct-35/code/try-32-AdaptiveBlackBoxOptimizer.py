import random
import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    """
    An optimization algorithm that uses adaptive search strategies to find the optimal solution.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        """
        Optimize the black box function using the given budget for function evaluations.
        
        Parameters:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        # Initialize the search space
        lower_bound = -5.0
        upper_bound = 5.0
        
        # Initialize the best solution and its cost
        best_solution = None
        best_cost = float('inf')
        
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Initialize the current solution
            new_individual = np.random.uniform(-5.0, 5.0, self.dim)
            
            # Evaluate the function at the current solution
            cost = func(new_individual)
            
            # If the current solution is better than the best solution found so far, update the best solution
            if cost < best_cost:
                best_solution = new_individual
                best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

    def optimize(self, func, bounds):
        """
        Optimize the function using a specified search strategy.
        
        Parameters:
        func (function): The black box function to optimize.
        bounds (list): A list of tuples containing the lower and upper bounds for each dimension.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        # Perform the optimization using differential evolution
        result = differential_evolution(func, bounds)
        
        # Return the optimal solution and its cost
        return result.x, result.fun

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Search Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer.optimize(func, [-5.0, 5.0])
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost
# 
# def main():
#     budget = 1000
#     dim = 10
#     best_solution, best_cost = black_box_optimizer(budget, dim)
#     print("Optimal solution:", best_solution)
#     print("Optimal cost:", best_cost)
# 
# if __name__ == "__main__":
#     main()

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Search Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import time
# import matplotlib.pyplot as plt

def adaptive_black_box_optimizer(budget, dim):
    """
    Optimize the function using a specified search strategy.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    
    Returns:
    tuple: A tuple containing the optimal solution and its cost.
    """
    # Initialize the search space
    lower_bound = -5.0
    upper_bound = 5.0
    
    # Initialize the best solution and its cost
    best_solution = None
    best_cost = float('inf')
    
    # Initialize the population size
    population_size = 100
    
    # Initialize the population
    population = [[np.random.uniform(lower_bound, upper_bound, dim) for _ in range(population_size)] for _ in range(population_size)]
    
    # Initialize the fitness function
    def fitness(individual):
        # Evaluate the function at the current individual
        cost = func(individual)
        
        # Return the fitness value
        return -cost
    
    # Perform the optimization
    for _ in range(budget):
        # Select the next generation
        next_generation = []
        for _ in range(population_size):
            # Select two parents using tournament selection
            parent1, parent2 = random.sample(population, 2)
            
            # Evaluate the fitness of the parents
            fitness1 = fitness(parent1)
            fitness2 = fitness(parent2)
            
            # Select the child using tournament selection
            child = (parent1[0] + parent2[0]) / 2
            
            # Evaluate the fitness of the child
            fitness_child = fitness(child)
            
            # Add the child to the next generation if it has a higher fitness
            if fitness_child > fitness1 + fitness2:
                next_generation.append(child)
            else:
                next_generation.append(parent1)
        
        # Replace the old population with the new generation
        population = next_generation
    
    # Return the optimal solution and its cost
    return population[0], fitness(population[0])

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Search Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import time
# import matplotlib.pyplot as plt

def main():
    budget = 1000
    dim = 10
    best_solution, best_cost = adaptive_black_box_optimizer(budget, dim)
    print("Optimal solution:", best_solution)
    print("Optimal cost:", best_cost)
    
    # Plot the convergence curve
    x = np.linspace(-5.0, 5.0, 100)
    y = np.array([best_cost for _ in range(len(x))])
    plt.plot(x, y)
    plt.xlabel("Dimension")
    plt.ylabel("Cost")
    plt.title("Convergence Curve")
    plt.show()

if __name__ == "__main__":
    main()