import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.best_solution = None
        self.best_score = -np.inf

    def __call__(self, func):
        # Initialize population with random solutions
        self.population = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(100)]

        # Evaluate population and select best solution
        for _ in range(self.budget):
            # Evaluate population
            scores = [self.evaluate(func, solution) for solution in self.population]

            # Select best solution
            best_index = np.argmax(scores)
            best_solution = self.population[best_index]

            # Update best solution and score
            self.best_solution = best_solution
            self.best_score = scores[best_index]

            # Refine the population
            self.population = [func(solution) for solution in self.population[best_index + 1:]]

            # If the best score is better than the current best score, update the best solution and score
            if self.best_score > self.best_score:
                self.best_solution = best_solution
                self.best_score = self.best_score

        return self.best_solution

    def evaluate(self, func, solution):
        return func(solution)

# Define the Black Box Optimization problem
def func1(solution):
    return np.sum(solution ** 2)

def func2(solution):
    return np.prod(solution)

def func3(solution):
    return np.mean(solution)

# Create an instance of the BlackBoxOptimizer class
optimizer = BlackBoxOptimizer(100, 10)

# Call the optimizer function
best_solution = optimizer(func1)

# Print the result
print(f"Best solution: {best_solution}")
print(f"Best score: {optimizer.best_score}")