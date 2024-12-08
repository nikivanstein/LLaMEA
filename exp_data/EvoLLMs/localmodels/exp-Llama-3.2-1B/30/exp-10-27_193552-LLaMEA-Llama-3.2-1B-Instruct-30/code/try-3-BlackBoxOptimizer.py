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
        self.perturbation_threshold = 0.3  # threshold for adaptive perturbation

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
        
        # Update the perturbation threshold
        self.perturbation_threshold = max(0, self.perturbation_threshold - 0.01 * (solution[0] - self.search_space[0][0]) / (self.search_space[0][1] - self.search_space[0][0]))
        
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
        
        # Refine the solution using adaptive perturbation
        if self.func_evaluations > self.budget:
            perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
            solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
        return solution, cost

# Description: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# Code: 
# ```python
# Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# ```
# ```python
# ```python
# import random
# import numpy as np
# import matplotlib.pyplot as plt

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
        self.perturbation_threshold = 0.3  # threshold for adaptive perturbation

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
        
        # Update the perturbation threshold
        self.perturbation_threshold = max(0, self.perturbation_threshold - 0.01 * (solution[0] - self.search_space[0][0]) / (self.search_space[0][1] - self.search_space[0][0]))
        
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
        
        # Refine the solution using adaptive perturbation
        if self.func_evaluations > self.budget:
            perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
            solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
        return solution, cost

# Description: Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# Code: 
# ```python
# Randomized Black Box Optimization Algorithm with Adaptive Perturbation
# ```
# ```python
# import random
# import numpy as np
# import matplotlib.pyplot as plt

def evaluate_bbof(func, x_range):
    """
    Evaluate the black box function at the given range of x values.
    
    Args:
        func (function): The black box function to evaluate.
        x_range (tuple): The range of x values to evaluate.
    
    Returns:
        float: The cost of the optimal solution.
    """
    
    # Evaluate the black box function at the given range of x values
    x_values = np.linspace(x_range[0], x_range[1], 100)
    y_values = func(x_values)
    
    # Return the cost of the optimal solution
    return np.min(y_values)

# Generate a random function
def random_function(x):
    """
    Generate a random function.
    
    Args:
        x (float): The input value.
    
    Returns:
        float: The output value of the random function.
    """
    
    # Generate a random function
    return np.random.uniform(-5.0, 5.0)

# Run the optimizer
optimizer = BlackBoxOptimizer(budget=100, dim=10)
solution, cost = optimizer(func=random_function, num_iterations=1000)
print(f"Optimal solution: {solution}")
print(f"Cost: {cost}")

# Refine the solution using adaptive perturbation
optimizer = BlackBoxOptimizer(budget=100, dim=10)
solution, cost = optimizer(func=random_function, num_iterations=1000)
print(f"Optimal solution: {solution}")
print(f"Cost: {cost}")

# Plot the results
x_range = (-5.0, 5.0)
y_values = evaluate_bbof(random_function, x_range)
plt.plot(x_range, y_values)
plt.show()