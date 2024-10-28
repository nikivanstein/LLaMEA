import numpy as np
import random
import copy

class DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.g = None
        self.m = None
        self.m_history = []
        self.x_history = []
        self.population_size = 100
        self.tournament_size = 5
        self.crossover_probability = 0.7
        self.mutation_probability = 0.1

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Insufficient budget")

        # Initialize the current solution
        self.x = np.random.uniform(-5.0, 5.0, self.dim)
        self.f = func(self.x)

        # Initialize the mutation rate
        self.m = 0.1

        while self.budget > 0:
            # Evaluate the function at the current solution
            self.f = func(self.x)

            # Generate a new solution using differential evolution
            self.g = self.x + np.random.normal(0, 1, self.dim) * np.sqrt(self.f / self.budget)
            self.g = np.clip(self.g, -5.0, 5.0)

            # Evaluate the new solution
            self.g = func(self.g)

            # Check if the new solution is better
            if self.f < self.g:
                # Select two parents using tournament selection
                parents = self.select_parents(self.x, self.g)

                # Create a new individual by crossover and mutation
                self.x = self.crossover(parents[0], parents[1])
                self.m = self.mutation(parents[0], parents[1], self.m)

                # Update the mutation rate
                self.m = self.m / 2

            # Update the history
            self.x_history.append(self.x)
            self.m_history.append(self.m)

            # Decrease the budget
            self.budget -= 1

            # Check if the budget is zero
            if self.budget == 0:
                break

        return self.x

    def select_parents(self, parent1, parent2):
        # Select two parents using tournament selection
        parent1_fitness = np.sum(parent1**2)
        parent2_fitness = np.sum(parent2**2)
        tournament = np.random.randint(0, self.population_size, self.tournament_size)
        selected_parents = [parent1, parent2]
        for i in tournament:
            if self.f(parent1) > self.f(parent2):
                selected_parents[i] = parent1
            else:
                selected_parents[i] = parent2
        return selected_parents

    def crossover(self, parent1, parent2):
        # Perform crossover
        crossover_point = np.random.randint(0, self.dim)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, parent1, parent2, mutation_rate):
        # Perform mutation
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(0, self.dim)
            parent1[mutation_point] += np.random.normal(0, 1)
            parent2[mutation_point] += np.random.normal(0, 1)
        return parent1, parent2

# Example usage:
def test_func(x):
    return np.sum(x**2)

de = DE(1000, 10)
opt_x = de(__call__, test_func)
print(opt_x)