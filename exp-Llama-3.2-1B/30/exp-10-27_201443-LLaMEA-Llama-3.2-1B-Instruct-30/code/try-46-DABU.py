import numpy as np
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def refine_strategy(self, func, initial_solution, budget):
        # Initialize the population with random solutions
        population = [initial_solution]
        for _ in range(budget):
            # Evaluate the fitness of each solution and select the fittest one
            fitnesses = [func(solution) for solution in population]
            fittest_index = np.argmin(fitnesses)
            new_solution = population[fittest_index]
            # Apply a mutation strategy to introduce randomness
            mutation_rate = 0.01
            if random.random() < mutation_rate:
                new_solution[0] += random.uniform(-1, 1)
                new_solution[1] += random.uniform(-1, 1)
            # Add the new solution to the population
            population.append(new_solution)
        # Select the fittest solution from the new population
        fittest_solution = population[np.argmax([func(solution) for solution in population])]
        return fittest_solution, budget - self.func_evaluations

def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

def bbopt(func, initial_solution, budget):
    return DABU(budget, len(initial_solution)).refine_strategy(func, initial_solution, budget)

# Example usage:
initial_solution = np.array([-2, -2])
fittest_solution, budget = bbopt(test_function, initial_solution, 1000)
print(f"Fittest solution: {fittest_solution}")
print(f"Fitness: {test_function(fittest_solution)}")