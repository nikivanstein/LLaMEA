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
        self.perturbation_step_size = 0.1
    
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
        population = self.initialize_population(func, num_iterations)
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individual
            fittest_individual = self.select_fittest(population)
            
            # Create a new population by perturbing the fittest individual
            new_population = self.perturb_population(population, fittest_individual)
            
            # Evaluate the new population
            new_cost = self.evaluate_population(func, new_population)
            
            # Update the population
            population = new_population
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return self.evaluate_population(func, population)

    def initialize_population(self, func, num_iterations):
        """
        Initialize the population with random solutions.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            list: The initialized population.
        """
        
        # Initialize the population with random solutions
        population = []
        for _ in range(num_iterations):
            # Generate a random solution
            individual = (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
            
            # Evaluate the solution
            fitness = func(individual)
            
            # Add the solution to the population
            population.append((individual, fitness))
        
        return population
    
    def perturb_population(self, population, fittest_individual):
        """
        Perturb the population by creating new individuals.
        
        Args:
            population (list): The population to perturb.
            fittest_individual (tuple): The fittest individual.
        
        Returns:
            list: The perturbed population.
        """
        
        # Initialize the new population with the fittest individual
        new_population = population.copy()
        
        # Perturb the new population
        for _ in range(len(population)):
            # Select a random individual from the population
            individual = random.choice(population)
            
            # Perturb the individual
            perturbed_individual = (individual[0] + random.uniform(-self.perturbation_step_size, self.perturbation_step_size), 
                                 individual[1] + random.uniform(-self.perturbation_step_size, self.perturbation_step_size))
            
            # Add the perturbed individual to the new population
            new_population.append((perturbed_individual, func(perturbed_individual)))
        
        return new_population
    
    def evaluate_population(self, func, population):
        """
        Evaluate the population using the given function.
        
        Args:
            func (function): The black box function to optimize.
            population (list): The population to evaluate.
        
        Returns:
            float: The cost of the population.
        """
        
        # Initialize the cost
        cost = 0
        
        # Evaluate the population
        for individual, fitness in population:
            cost += fitness
        
        return cost
    
    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of an individual using the given function.
        
        Args:
            individual (tuple): The individual to evaluate.
            func (function): The black box function to optimize.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the function at the individual
        fitness = func(individual)
        
        return fitness

# Description: Randomized Black Box Optimization Algorithm
# Code: 
# ```python
# import random
# import numpy as np
# import blackbox_optimizer as bbo

# class RandomizedBlackBoxOptimizer(bbo.BlackBoxOptimizer):
#     """
#     A metaheuristic algorithm to optimize black box functions.
#     """
#     def __init__(self, budget, dim):
#         """
#         Initialize the optimizer with a budget and dimensionality.
#         """
#         super().__init__(budget, dim)
        
#     def __call__(self, func):
#         """
#         Optimize the black box function using the optimizer.
#         """
#         solution, cost = super().__call__(func)
#         return solution, cost

# # Test the optimizer
# bbo_optimizer = RandomizedBlackBoxOptimizer(100, 10)
# func = lambda x: np.sin(x)
# solution, cost = bbo_optimizer(func, 1000)
# print(f"Optimal solution: {solution}, Cost: {cost}")