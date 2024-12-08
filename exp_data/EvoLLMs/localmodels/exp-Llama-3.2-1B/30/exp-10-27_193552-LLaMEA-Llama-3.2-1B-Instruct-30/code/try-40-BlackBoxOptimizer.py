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
        self.perturbation_factor = 0.3
        self.perturbation_threshold = 1.0
        
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
        
        # Adjust the perturbation factor
        if random.random() < self.perturbation_factor:
            perturbation = (perturbation[0] + random.uniform(-self.perturbation_threshold, self.perturbation_threshold), 
                            perturbation[1] + random.uniform(-self.perturbation_threshold, self.perturbation_threshold))
        
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
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return solution, cost

# Description: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# Code: 
# ```python
# BlackBoxOptimizer: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# 
# Optimizes black box functions using a combination of random search and perturbation.
# The perturbation factor and threshold are used to adapt the perturbation strategy.
# 
# Parameters:
#     budget (int): The maximum number of function evaluations allowed.
#     dim (int): The dimensionality of the search space.
# 
# Returns:
#     tuple: The optimal solution and its cost.
# """
# 
# ```python
# import random
# import numpy as np

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
        self.perturbation_factor = 0.3
        self.perturbation_threshold = 1.0
        
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
        
        # Adjust the perturbation factor
        if random.random() < self.perturbation_factor:
            perturbation = (perturbation[0] + random.uniform(-self.perturbation_threshold, self.perturbation_threshold), 
                            perturbation[1] + random.uniform(-self.perturbation_threshold, self.perturbation_threshold))
        
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
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return solution, cost

# Description: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# Code: 
# ```python
# BlackBoxOptimizer: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# 
# Optimizes black box functions using a combination of random search and perturbation.
# The perturbation factor and threshold are used to adapt the perturbation strategy.
# 
# Parameters:
#     budget (int): The maximum number of function evaluations allowed.
#     dim (int): The dimensionality of the search space.
# 
# Returns:
#     tuple: The optimal solution and its cost.
# """
# 
# ```python
# import random
# import numpy as np

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
        self.perturbation_factor = 0.3
        self.perturbation_threshold = 1.0
        
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
        
        # Adjust the perturbation factor
        if random.random() < self.perturbation_factor:
            perturbation = (perturbation[0] + random.uniform(-self.perturbation_threshold, self.perturbation_threshold), 
                            perturbation[1] + random.uniform(-self.perturbation_threshold, self.perturbation_threshold))
        
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
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return solution, cost

# Description: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# Code: 
# ```python
# BlackBoxOptimizer: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# 
# Optimizes black box functions using a combination of random search and perturbation.
# The perturbation factor and threshold are used to adapt the perturbation strategy.
# 
# Parameters:
#     budget (int): The maximum number of function evaluations allowed.
#     dim (int): The dimensionality of the search space.
# 
# Returns:
#     tuple: The optimal solution and its cost.
# """
# 
# ```python
# import random
# import numpy as np

def main():
    # Create a new optimizer with a budget of 100 and a dimensionality of 10
    optimizer = BlackBoxOptimizer(budget=100, dim=10)
    
    # Define a black box function to optimize
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Run the optimizer for 10 iterations
    solution, cost = optimizer.run(func, num_iterations=10)
    
    # Print the optimal solution and its cost
    print(f"Optimal solution: {solution}")
    print(f"Cost: {cost}")
    
    # Update the optimizer with a new perturbation factor and threshold
    optimizer.perturbation_factor = 0.4
    optimizer.perturbation_threshold = 1.2
    
    # Run the optimizer for another 10 iterations
    solution, cost = optimizer.run(func, num_iterations=10)
    
    # Print the new optimal solution and its cost
    print(f"New optimal solution: {solution}")
    print(f"New cost: {cost}")

if __name__ == "__main__":
    main()