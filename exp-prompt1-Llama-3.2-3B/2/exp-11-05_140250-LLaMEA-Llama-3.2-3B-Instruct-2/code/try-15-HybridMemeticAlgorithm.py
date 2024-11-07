import numpy as np
import random

class HybridMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.memory = []
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.pbest_count = np.zeros((self.population_size, self.dim))
        self.random_solution = np.random.uniform(-5.0, 5.0, size=self.dim)
        self.swarm_size = int(self.population_size * 0.2)
        self.particles = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

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
                mutation = np.random.uniform(-0.1, 0.1, size=self.dim)
                mutated_solution = random_solution + mutation
                # Apply crossover
                crossover = np.random.choice([0, 1], size=self.dim)
                if crossover[0] == 1:
                    mutated_solution[crossover[1]] = random_solution[crossover[1]]
                # Replace the solution with the mutated solution
                population[i, :] = mutated_solution
            # Evaluate the population again
            evaluations = func(population)
            # Update the population with the best solutions
            for i in range(self.population_size):
                if evaluations[i] < evaluations[self.pbest_count[i, :]]:
                    self.pbest[i, :] = population[i, :]
                    self.pbest_count[i, :] = i
            # Update the particles
            for j in range(self.swarm_size):
                # Evaluate the particle
                evaluation = func(self.particles[j, :])
                # Update the particle
                self.particles[j, :] = self.particles[j, :] + 0.5 * (self.particles[j, :] - self.gbest)
                # Update the personal best
                if evaluation < evaluations[self.pbest_count[j, :]]:
                    self.pbest[j, :] = self.particles[j, :]
                    self.pbest_count[j, :] = j
            # Update the global best
            min_evaluation = np.min(evaluations)
            if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                self.gbest = evaluations[np.argmin(self.gbest)]
        return self.gbest