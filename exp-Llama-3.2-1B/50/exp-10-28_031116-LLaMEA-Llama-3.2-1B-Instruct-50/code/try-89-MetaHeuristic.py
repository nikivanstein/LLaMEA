import numpy as np
import random

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.init_population()
        self.fitness_scores = self.init_fitness_scores()

    def init_population(self):
        # Initialize population with random solutions within the search space
        population = []
        for _ in range(100):  # Number of initial solutions
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def init_fitness_scores(self):
        # Initialize fitness scores for each solution
        fitness_scores = []
        for solution in self.population:
            func_value = self.evaluate_func(solution)
            fitness_scores.append(func_value)
        return fitness_scores

    def evaluate_func(self, solution):
        # Evaluate the black box function at the given solution
        func_value = self.func(solution)
        return func_value

    def __call__(self, func):
        # Optimize the black box function using the given function
        while len(self.population) < self.budget:
            # Select a random parent from the population
            parent = random.choice(self.population)

            # Generate two new solutions by perturbing the parent
            child1 = parent + random.uniform(-self.budget, self.budget)
            child2 = parent - random.uniform(-self.budget, self.budget)

            # Evaluate the fitness of each child solution
            fitness1 = self.evaluate_func(child1)
            fitness2 = self.evaluate_func(child2)

            # Select the better child solution based on the probability of 0.45
            if random.random() < 0.45:
                child1 = child2
            elif random.random() < 0.9:
                child1 = parent

            # Update the population with the selected child solution
            self.population.append(child1)
            self.population.append(child2)

            # Update the fitness scores
            self.fitness_scores.append(fitness1)
            self.fitness_scores.append(fitness2)

        # Select the best solution from the final population
        best_solution = self.population[np.argmax(self.fitness_scores)]
        return best_solution

    def get_solution(self, func):
        # Get the best solution from the population
        best_solution = self.population[np.argmax(self.fitness_scores)]
        return best_solution