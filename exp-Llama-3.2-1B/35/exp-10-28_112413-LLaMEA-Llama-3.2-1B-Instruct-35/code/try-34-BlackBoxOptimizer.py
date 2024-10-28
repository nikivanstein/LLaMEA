import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer.

        Parameters:
        budget (int): The number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.x_history = []

    def __call__(self, func, mutation_prob, mutation_rate):
        """
        Optimize the black box function using the given budget.

        Parameters:
        func (function): The black box function to optimize.
        mutation_prob (float): The probability of mutating an individual.
        mutation_rate (float): The rate at which an individual is mutated.

        Returns:
        float: The optimized value of the function.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Evaluate the function at the current point
            y = func(x)
            
            # Update the search space if the current function value is better
            if y > x[-1]:
                # Refine the search space using Stochastic Gradient Descent
                new_x = stgd(x, func, self.epsilon, self.learning_rate)
                
                # If mutation is allowed, mutate the new individual
                if np.random.rand() < mutation_prob:
                    new_x = mutation(new_x, mutation_rate)
                
                # Update the search space
                x = new_x
                
                # Update the history
                self.x_history.append(x)
        
        # Return the optimized value of the function
        return x[-1]

def stgd(x, func, epsilon, learning_rate):
    """
    Refine the search space using Stochastic Gradient Descent.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    epsilon (float): The step size for the gradient descent update.
    learning_rate (float): The step size for the gradient descent update.

    Returns:
    numpy array: The updated point in the search space.
    """
    y = func(x)
    grad = (y - x[-1]) / epsilon
    x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
    return x

def func(x):
    return x**2

# Initialize the optimizer
optimizer = BlackBoxOptimizer(1000, 10)
# Evaluate the fitness of the initial individual
optimized_x = optimizer(func, 0.35, 0.5)
print(optimized_x)

# Print the history of the search space
print("Search space history:")
for i, x in enumerate(optimizer.x_history):
    print(f"Individual {i+1}: {x}")