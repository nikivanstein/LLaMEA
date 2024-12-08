# Description: Novel MGDALR Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func, initial_individual=None):
        def inner(individual):
            return func(individual)
        
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
        
        # If no initial individual is provided, use the current x as the initial individual
        if initial_individual is None:
            initial_individual = x
        
        return x

# Test the algorithm
def test_mgdalr():
    func = lambda x: x[0]**2 + x[1]**2
    initial_individual = np.array([-2.0, -2.0])
    solution = MGDALR(100, 2).call(func, initial_individual)
    print("Solution:", solution)
    print("Score:", func(solution))

# Test the algorithm with an initial individual
def test_mgdalr_initial():
    func = lambda x: x[0]**2 + x[1]**2
    solution = MGDALR(100, 2).call(func, np.array([-2.0, -2.0]))
    print("Solution:", solution)
    print("Score:", func(solution))

# Evaluate the algorithm using the BBOB test suite
def evaluate_mgdalr(func, num_evaluations=100):
    # Create a dictionary to store the results
    results = {}
    
    # Loop through each function in the BBOB test suite
    for name, func in func.items():
        # Initialize the population with the current individual
        population = [initial_individual] if initial_individual is not None else np.array([-2.0, -2.0])
        
        # Run the algorithm for the specified number of evaluations
        for _ in range(num_evaluations):
            # Get the current population
            current_population = population[:]
            
            # Run the algorithm for the specified number of iterations
            for _ in range(100):
                # Run the algorithm
                solution = MGDALR(100, 2).call(func, current_population)
                
                # Update the population
                current_population = [solution] + current_population[:-1]
        
        # Store the results
        results[name] = current_population[0]

# Run the evaluation
evaluate_mgdalr(func)

# Print the results
print("BBOB Test Suite Results:")
for name, score in results.items():
    print(f"{name}: {score}")