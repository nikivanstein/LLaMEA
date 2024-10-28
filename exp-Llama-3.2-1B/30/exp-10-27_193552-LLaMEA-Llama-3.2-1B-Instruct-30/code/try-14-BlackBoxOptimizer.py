# Description: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution.
    It handles a wide range of tasks and can be adapted to various optimization problems.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = [(-5.0, 5.0)] * dim
        self.func_evaluations = 0
        
    def __call__(self, func):
        """
        Optimize the black box function using the optimizer.
        
        Args:
            func (function): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the solution and cost
        solution = None
        cost = float('inf')
        
        # Perform random search
        for _ in range(self.budget):
            # Perturb the current solution
            perturbed_solution = self.perturb(solution)
            
            # Evaluate the new solution
            new_cost = func(perturbed_solution)
            
            # Update the solution and cost if the new solution is better
            if new_cost < cost:
                solution = perturbed_solution
                cost = new_cost
        
        return solution, cost
    
    def perturb(self, solution):
        """
        Perturb the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            tuple: The perturbed solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
        return solution
    
    def run(self, func, num_iterations):
        """
        Run the optimizer for a specified number of iterations.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the population with random solutions
        population = [self.evaluate_fitness(func) for _ in range(100)]  # Use a smaller population size for faster convergence
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individual
            fittest_individual = population.index(max(population))
            
            # Perturb the fittest individual
            perturbed_individual = self.perturb(population[fittest_individual])
            
            # Evaluate the new individual
            new_individual, new_cost = self.evaluate_fitness(perturbed_individual)
            
            # Update the population with the new individual
            population[fittest_individual] = perturbed_individual
            
            # Update the fittest individual
            fittest_individual = population.index(max(population))
            
            # Update the cost if the new individual is better
            if new_cost < population[fittest_individual]:
                population[fittest_individual] = new_individual
        
        return population[0], population[0]

    def evaluate_fitness(self, func):
        """
        Evaluate the fitness of a given function.
        
        Args:
            func (function): The black box function to evaluate.
        
        Returns:
            float: The fitness of the function.
        """
        
        # Generate a random input
        input = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Evaluate the function with the input
        fitness = func(input)
        
        return fitness
    
    def update_perturbation(self, perturbation):
        """
        Update the perturbation with a probability based on the fitness of the current individual.
        
        Args:
            perturbation (tuple): The current perturbation.
        
        Returns:
            tuple: The updated perturbation.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the perturbation with a probability based on the fitness of the current individual
        if random.random() < 0.3:
            perturbation = (perturbation[0] + 0.1 * (perturbation[0] - random.uniform(-5.0, 5.0)), perturbation[1] + 0.1 * (perturbation[1] - random.uniform(-5.0, 5.0)))
        
        return perturbation

# Description: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# Code: 
# ```python
# import random
# import numpy as np
# import matplotlib.pyplot as plt

# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = [(-5.0, 5.0)] * dim
#         self.func_evaluations = 0
        
#     def __call__(self, func):
#         solution = None
#         cost = float('inf')
#         population = [self.evaluate_fitness(func) for _ in range(100)]
        
#         for _ in range(self.budget):
#             fittest_individual = population.index(max(population))
#             perturbed_individual = self.perturb(population[fittest_individual])
#             new_individual, new_cost = self.evaluate_fitness(perturbed_individual)
#             population[fittest_individual] = perturbed_individual
#             fittest_individual = population.index(max(population))
#             if new_cost < population[fittest_individual]:
#                 population[fittest_individual] = new_individual
        
#         return population[0], population[0]
        
#     def perturb(self, solution):
#         perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
#         solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
#     def run(self, func, num_iterations):
#         population = [self.evaluate_fitness(func) for _ in range(100)]
        
#         for _ in range(num_iterations):
#             fittest_individual = population.index(max(population))
#             perturbed_individual = self.update_perturbation(population[fittest_individual])
#             new_individual, new_cost = self.evaluate_fitness(perturbed_individual)
#             population[fittest_individual] = perturbed_individual
        
#         return population[0], population[0]
        
# def main():
#     optimizer = BlackBoxOptimizer(100, 5)
#     func = lambda x: x**2
#     solution, cost = optimizer(100, 100)
#     print("Optimal solution:", solution)
#     print("Cost:", cost)
#     plt.plot([solution, func(solution)])
#     plt.show()

# if __name__ == "__main__":
#     main()