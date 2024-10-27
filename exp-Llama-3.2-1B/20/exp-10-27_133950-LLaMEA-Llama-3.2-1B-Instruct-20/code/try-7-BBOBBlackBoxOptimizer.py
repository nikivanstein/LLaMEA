import numpy as np
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def adapt_strategy(self, individual, fitness):
        # Refine strategy based on fitness and probability
        probability = 0.2
        if fitness < 0.5:
            # Increase exploration by 20%
            self.search_space = np.linspace(-5.0, 5.0, 200)
            individual = self.evaluate_fitness(individual, self.logger)
        elif fitness > 0.5:
            # Decrease exploration by 20%
            self.search_space = np.linspace(-5.0, 5.0, 80)
            individual = self.evaluate_fitness(individual, self.logger)
        else:
            # Maintain current strategy
            pass

        # Update individual based on new strategy
        individual = self.evaluate_fitness(individual, self.logger)

        return individual

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)

# Print initial solution
print("Initial Solution:")
print(result)

# Print updated solution after 1000 evaluations
print("\nUpdated Solution after 1000 evaluations:")
print(optimizer(func))

# Refine strategy based on fitness
optimized_individual = optimizer.func_evaluations
optimized_func = lambda x: x**2
optimized_result = optimizer(optimized_func)

# Print final solution
print("\nFinal Solution:")
print(optimized_result)