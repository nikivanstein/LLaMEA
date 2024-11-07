import numpy as np
import random

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.memory = []
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.pbest_count = np.zeros((self.population_size, self.dim))
        self.random_solution = np.random.uniform(-5.0, 5.0, size=self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a new population
            population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Evaluate the population
            evaluations = func(population)
            # Update the population with the best solutions
            for i in range(self.population_size):
                if evaluations[i] < evaluations[self.pbest_count[i, :]]:
                    self.pbest[i, :] = population[i, :]
                    self.pbest_count[i, :] = i
            # Update the global best solution
            min_evaluation = np.min(evaluations)
            if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                self.gbest = evaluations[np.argmin(self.gbest)]
            # Update the memory with the best solutions
            self.memory.append(self.pbest[self.pbest_count == i, :])
            # Apply memetic operators
            for i in range(self.population_size):
                # Select a random solution from the memory
                random_solution = np.random.choice(self.memory, size=1)[0]
                # Mutate the solution
                mutation = np.random.uniform(-0.05, 0.05, size=self.dim) # Changed to (0.05)
                mutated_solution = random_solution + mutation
                # Apply crossover
                crossover = np.random.choice([0, 1], size=self.dim)
                if crossover[0] == 1:
                    mutated_solution[crossover[1]] = random_solution[crossover[1]]
                # Replace the solution with the mutated solution
                population[i, :] = mutated_solution
                # Apply a new mutation operator
                if np.random.rand() < 0.02: # Changed to 0.02 (2% chance)
                    mutation = np.random.uniform(-0.1, 0.1, size=self.dim)
                    mutated_solution = random_solution + mutation
                # Evaluate the population again
                evaluations = func(population)
                # Update the population with the best solutions
                for j in range(self.population_size):
                    if evaluations[j] < evaluations[self.pbest_count[j, :]]:
                        self.pbest[j, :] = population[j, :]
                        self.pbest_count[j, :] = j
                # Update the global best solution
                min_evaluation = np.min(evaluations)
                if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                    self.gbest = evaluations[np.argmin(self.gbest)]
        return self.gbest