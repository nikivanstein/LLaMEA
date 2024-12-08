import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.population_size_mutations = 0
        self.mutations = 0
        self.tournament_size = 5
        self.tournament_size_mutations = 0
        self.pareto_front = None

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
        if random.random() < self.mutations / self.population_size:
            # Randomly select two points in the individual
            point1, point2 = random.sample(range(self.dim), 2)
            # Swap the points
            individual[point1], individual[point2] = individual[point2], individual[point1]
            # Increment the population size mutations
            self.population_size_mutations += 1
        if random.random() < self.population_size_mutations / self.population_size:
            # Randomly select a mutation point
            point = random.randint(0, self.dim - 1)
            # Flip the bit at the mutation point
            individual[point] = 1 - individual[point]
            # Increment the population size mutations
            self.population_size_mutations += 1

    def tournament(self, individuals):
        # Select the best individual from the tournament
        best_individual = max(individuals, key=lambda individual: individual[0])
        # Return the best individual
        return best_individual

    def get_pareto_front(self, num_frontiers):
        # Generate a random Pareto front
        frontiers = np.random.uniform(self.search_space[0], self.search_space[1], size=(num_frontiers, self.dim))
        # Sort the frontiers in non-dominated order
        sorted_frontiers = sorted(zip(frontiers, np.random.uniform(self.search_space[0], self.search_space[1], size=(num_frontiers, self.dim))), key=lambda pair: pair[1], reverse=True)
        # Return the non-dominated frontiers
        return [pair[0] for pair in sorted_frontiers]

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"