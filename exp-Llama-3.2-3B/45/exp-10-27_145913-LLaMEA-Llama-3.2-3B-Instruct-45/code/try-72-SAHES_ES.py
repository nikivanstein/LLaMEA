import numpy as np
import random

class SAHES_ES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_size = 50
        self.population_size = 100
        self.best_solution = None
        self.best_value = float('inf')
        self.refinement_probability = 0.45

    def __call__(self, func):
        if self.budget == 0:
            return self.best_value

        if self.best_solution is None:
            # Initialize the harmony memory with random solutions
            self.harmony = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.harmony_size, self.dim))
            # Evaluate the harmony memory
            self.evaluate_harmony(func)

        for _ in range(self.budget):
            # Create a new solution using the harmony memory
            new_solution = self.create_new_solution(self.harmony)
            # Evaluate the new solution
            value = func(new_solution)
            # Update the best solution if necessary
            if value < self.best_value:
                self.best_solution = new_solution
                self.best_value = value
            # Update the harmony memory if necessary
            if self.evaluate_harmony(func, value):
                self.harmony = self.update_harmony(self.harmony, new_solution, value)
            # Refine the strategy with probability 0.45
            if random.random() < self.refinement_probability:
                self.refine_strategy(self.harmony, self.best_solution, value)
            # Update the population size if necessary
            if len(self.harmony) < self.population_size:
                self.harmony = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.population_size, self.dim))
                self.evaluate_harmony(func)

        return self.best_value

    def evaluate_harmony(self, func):
        values = func(self.harmony)
        # Select the best solutions
        self.harmony = self.harmony[np.argsort(values)]
        # Update the best solution if necessary
        if values[0] < self.best_value:
            self.best_solution = self.harmony[0]
            self.best_value = values[0]

    def create_new_solution(self, harmony):
        # Create a new solution using the harmony memory
        new_solution = np.zeros(self.dim)
        for i in range(self.dim):
            # Select a random solution from the harmony memory
            index = random.randint(0, self.harmony_size - 1)
            new_solution[i] = harmony[index, i]
        return new_solution

    def update_harmony(self, harmony, new_solution, value):
        # Update the harmony memory
        harmony[self.harmony[:, 0] == value] = new_solution
        return harmony

    def refine_strategy(self, harmony, best_solution, value):
        # Refine the strategy by changing the individual lines of the best solution
        for i in range(self.dim):
            # Select a random solution from the harmony memory
            index = random.randint(0, self.harmony_size - 1)
            # Change the value of the current dimension
            harmony[index, i] = best_solution[i] + random.uniform(-1, 1)
            # Ensure the new value is within the bounds
            harmony[index, i] = np.clip(harmony[index, i], self.lower_bound, self.upper_bound)
        # Evaluate the refined harmony memory
        values = func(self.harmony)
        # Update the best solution if necessary
        if values[0] < self.best_value:
            self.best_solution = self.harmony[0]
            self.best_value = values[0]

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
optimizer = SAHES_ES(budget, dim)
best_value = optimizer(func)
print("Best value:", best_value)