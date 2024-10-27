import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.rates = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        self.selection_rate = 0.5

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        # Select two parents using the roulette wheel selection strategy
        parents = np.random.choice([individual, self.search_space[0], self.search_space[1]], size=2, p=self.rates)
        # Select a random mutation point within the search space
        mutation_point = np.random.randint(0, 2, size=self.dim)
        # Create a new individual by swapping the genes at the mutation point
        new_individual = parents[0][:self.dim][mutation_point] + parents[1][:self.dim][mutation_point]
        return new_individual

    def select_parents(self, num_parents):
        # Select parents using the tournament selection strategy
        parents = []
        for _ in range(num_parents):
            # Get the top N parents
            top_parents = np.argsort(np.random.randint(0, len(self.func_evaluations), size=N, dtype=int))[:len(self.func_evaluations)//2]
            # Select the top parents
            parents.append(self.func_evaluations[top_parents].item())
        # Return the selected parents
        return parents

    def __str__(self):
        return "BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization"