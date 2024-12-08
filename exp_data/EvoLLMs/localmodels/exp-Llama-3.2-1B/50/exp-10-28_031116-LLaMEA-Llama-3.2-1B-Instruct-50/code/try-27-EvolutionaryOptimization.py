import numpy as np
import random

class EvolutionaryOptimization:
    def __init__(self, budget, dim, num_iterations=100, num_evaluations=10):
        self.budget = budget
        self.dim = dim
        self.num_iterations = num_iterations
        self.num_evaluations = num_evaluations
        self.population = self.initialize_population()
        self.best_individual = self.population[0]
        self.best_score = -np.inf

    def initialize_population(self):
        # Initialize population with random solutions
        population = []
        for _ in range(self.num_evaluations):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Evaluate the black box function
        func_values = [func(x) for x in self.population]
        # Select the best individual based on the budget
        selected_individuals = random.sample(self.population, min(self.num_evaluations, self.budget))
        # Initialize the best solution
        best_solution = selected_individuals[0]
        best_score = np.mean(func_values)
        # Evolve the population
        for _ in range(self.num_iterations):
            # Calculate the average function value
            avg_func_value = np.mean(func_values)
            # Refine the best solution based on the probability of success
            if random.random() < 0.45:
                best_solution = np.random.uniform(-5.0, 5.0, self.dim)
                best_score = np.mean(func_values)
            else:
                best_solution = best_solution + random.uniform(-0.1, 0.1, self.dim)
                best_score = np.mean(func_values)
            # Update the population
            self.population = selected_individuals + [best_solution]
            # Update the best solution
            if np.mean(func_values) > best_score:
                best_solution = best_solution
                best_score = np.mean(func_values)
        return best_solution, best_score

# Test the algorithm
def test_optimization():
    optimization = EvolutionaryOptimization(budget=100, dim=5)
    func = lambda x: np.sin(x)
    best_solution, best_score = optimization(__call__, func)
    print("Best solution:", best_solution)
    print("Best score:", best_score)
    print("Score of the best solution:", best_score)

# Run the test
test_optimization()