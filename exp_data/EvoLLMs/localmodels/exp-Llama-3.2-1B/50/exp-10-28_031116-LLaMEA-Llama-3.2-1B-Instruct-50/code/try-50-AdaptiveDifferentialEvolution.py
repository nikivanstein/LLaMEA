import numpy as np
import random

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, p1=0.9, p2=1.4, alpha=0.1):
        self.budget = budget
        self.dim = dim
        self.p1 = p1
        self.p2 = p2
        self.alpha = alpha
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Initialize the population with random solutions
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.budget)]
        self.fitness_scores = [func(self.population[i]) for i in range(self.budget)]

        # Perform ADE iterations
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = [self.select_parent(self.population[i], self.population[i+1]) for i in range(self.budget-1)]

            # Perform mutation
            self.population = [self.mutate(self.population[i]) for i in range(self.budget)]

            # Calculate fitness scores
            self.fitness_scores = [func(self.population[i]) for i in range(self.budget)]

            # Select next generation using adaptive probability of convergence
            if np.random.rand() < self.alpha:
                next_generation = [self.select_parent(parent, self.population[0]) for parent in parents]
            else:
                next_generation = [self.population[i] for i in range(self.budget)]

            # Update population and fitness scores
            self.population = next_generation
            self.fitness_scores = [func(self.population[i]) for i in range(self.budget)]

    def select_parent(self, parent1, parent2):
        # Select parent using tournament selection
        return random.choice([parent1, parent2])

    def mutate(self, solution):
        # Perform mutation using crossover and mutation operators
        if random.random() < self.p1:
            return solution[:random.randint(1, self.dim)]
        else:
            return solution[:random.randint(1, self.dim)] + solution[random.randint(1, self.dim):]

# One-line description with the main idea
# Adaptive Differential Evolution (ADE) algorithm for black box optimization
# The algorithm uses adaptive probability of convergence to refine its strategy