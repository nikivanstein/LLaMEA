import numpy as np
import random

class ProbabilisticMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.current_solution = None

    def __call__(self, func):
        # Initialize the population with random solutions
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(10)]

        # Evaluate the initial population
        self.evaluate_population(func)

        # Main optimization loop
        for _ in range(self.budget):
            # Select the best solution
            self.current_solution = self.select_best_solution(func)

            # Refine the best solution using probabilistic mutation
            self.refine_solution(func)

            # Evaluate the refined solution
            self.evaluate_solution(func)

    def evaluate_population(self, func):
        # Evaluate each solution in the population
        scores = [func(solution) for solution in self.population]
        self.population = list(zip(self.population, scores))

    def select_best_solution(self, func):
        # Select the best solution based on the fitness score
        best_solution = max(self.population, key=lambda x: x[1])
        return best_solution[0]

    def refine_solution(self, func):
        # Refine the best solution using probabilistic mutation
        if self.current_solution is not None:
            # Randomly select a dimension to mutate
            dim_to_mutate = random.randint(0, self.dim - 1)

            # Randomly decide whether to increase or decrease the value
            mutation_type = random.random()
            if mutation_type < 0.3:
                self.current_solution[dim_to_mutate] += random.uniform(0.0, 0.1)
            else:
                self.current_solution[dim_to_mutate] -= random.uniform(0.0, 0.1)

            # Ensure the value is within the bounds
            self.current_solution[dim_to_mutate] = max(-5.0, min(5.0, self.current_solution[dim_to_mutate]))

    def evaluate_solution(self, func):
        # Evaluate the refined solution
        score = func(self.current_solution)
        self.population = [(solution, score) for solution, score in self.population]
        self.population.sort(key=lambda x: x[1])