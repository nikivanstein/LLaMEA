import numpy as np
import random

class AEAP_E:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.best_solution = self.population[0]
        self.current_function_evaluations = 0

    def initialize_population(self):
        population = []
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append((solution, self.evaluate_function(solution)))
        return population

    def evaluate_function(self, func):
        return func()

    def __call__(self, func):
        if self.current_function_evaluations >= self.budget:
            return self.population

        for i in range(self.budget):
            if random.random() < 0.3:
                # Refine the current solution
                new_solution = self.population[i][0] + np.random.uniform(-0.1, 0.1, self.dim)
                self.population[i] = (new_solution, self.evaluate_function(new_solution))

        # Replace the worst solution with a new one
        self.population.sort(key=lambda x: x[1])
        self.population[0] = (self.population[i][0], self.evaluate_function(self.population[i][0]))
        self.population.sort(key=lambda x: x[1])

        self.best_solution = max(self.population, key=lambda x: x[1])
        self.current_function_evaluations += 1

        return self.population

# Example usage:
if __name__ == "__main__":
    # Define the black box function
    def func(x):
        return np.sum(x**2)

    # Create an instance of the AEAP_E algorithm
    aeap_e = AEAP_E(budget=100, dim=10)

    # Optimize the function using the AEAP_E algorithm
    optimized_population = aeap_e(func)

    # Print the optimized solution
    print("Optimized solution:", optimized_population[0])