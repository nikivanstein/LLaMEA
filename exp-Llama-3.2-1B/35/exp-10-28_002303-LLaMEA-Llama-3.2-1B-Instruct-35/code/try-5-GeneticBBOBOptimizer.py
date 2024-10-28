import random
import numpy as np

class GeneticBBOBOptimizer:
    """
    A novel metaheuristic algorithm that combines genetic principles with black box optimization using a population-based approach.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize the population with random individuals.
        
        Returns:
        list: A list of individuals.
        """
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def __call__(self, func):
        """
        Optimize the black box function using the given budget for function evaluations.
        
        Parameters:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Generate a new individual by mutating the current population
            new_individual = self.mutate(self.population)
            
            # Evaluate the function at the new individual
            cost = func(new_individual)
            
            # If the new individual is better than the best solution found so far, update the best solution
            if cost < self.best_solution_cost(new_individual):
                self.best_solution = new_individual
                self.best_solution_cost = cost
        
        # Return the optimal solution and its cost
        return self.best_solution, self.best_solution_cost

    def mutate(self, individual):
        """
        Mutate an individual by changing a random element with a probability of mutation_rate.
        
        Parameters:
        individual (list): The individual to mutate.
        
        Returns:
        list: The mutated individual.
        """
        if random.random() < self.mutation_rate:
            index = random.randint(0, len(individual) - 1)
            individual[index] = random.uniform(-5.0, 5.0)
        return individual

    def best_solution_cost(self, individual):
        """
        Calculate the cost of the best solution found so far.
        
        Parameters:
        individual (list): The best solution.
        
        Returns:
        float: The cost of the best solution.
        """
        return sum([func(individual[i]) for i in range(self.dim)])

# Description: GeneticBBOBOptimizer
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = GeneticBBOBOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
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