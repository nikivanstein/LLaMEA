import numpy as np

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

def optimize_func(func, budget, dim, exploration_rate=0.2):
    """
    Novel Metaheuristic Algorithm for Black Box Optimization

    Args:
        func (function): The black box function to optimize
        budget (int): The maximum number of function evaluations
        dim (int): The dimensionality of the search space
        exploration_rate (float, optional): The rate at which to explore the search space. Defaults to 0.2.

    Returns:
        tuple: The optimized individual and its fitness score
    """
    # Initialize the algorithm with the specified budget and dimension
    algorithm = MGDALR(budget, dim)
    
    # Initialize the current individual to the lower bound
    current_individual = np.array([-5.0] * dim)
    
    # Initialize the best individual and its score
    best_individual = current_individual
    best_score = -np.inf
    
    # Iterate until the budget is reached
    for _ in range(budget):
        # Evaluate the function at the current individual
        y = func(current_individual)
        
        # If we've reached the maximum number of iterations, stop exploring
        if exploration_rate < 1:
            break
        
        # If we've reached the upper bound, stop exploring
        if current_individual[-1] >= 5.0:
            break
        
        # Learn a new direction using gradient descent
        learning_rate = algorithm.learning_rate * (1 - algorithm.explore_rate / algorithm.max_explore_count)
        dx = -np.dot(current_individual - func(current_individual), np.gradient(y))
        current_individual += learning_rate * dx
        
        # Update the exploration count
        algorithm.explore_count += 1
        
        # If we've reached the upper bound, stop exploring
        if current_individual[-1] >= 5.0:
            break
        
        # Evaluate the function at the current individual
        y = func(current_individual)
        
        # Update the best individual and its score if necessary
        if y > best_score:
            best_individual = current_individual
            best_score = y
    
    # Return the optimized individual and its score
    return best_individual, best_score

# Example usage:
def test_func1(x):
    return np.sum(x**2)

def test_func2(x):
    return np.prod(x)

best_individual, best_score = optimize_func(test_func1, 100, 2)
print(f"Best individual: {best_individual}")
print(f"Best score: {best_score}")