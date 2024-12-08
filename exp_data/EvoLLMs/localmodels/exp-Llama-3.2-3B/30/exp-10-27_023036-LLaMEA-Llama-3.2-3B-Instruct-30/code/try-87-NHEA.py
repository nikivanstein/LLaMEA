import numpy as np
import random
import math

class NHEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.particle swarm parameters = {'r': 0.8, 'c': 0.4, 'w': 0.4}
        self.simulated annealing parameters = {'alpha': 0.99, 'T0': 1000, 'Tend': 1}

    def __call__(self, func):
        # Initialize the population
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

        # Evaluate the initial population
        fitness = [func(x) for x in population]

        # Main loop
        for _ in range(self.budget):
            # Select the fittest particles
            fittest_particles = np.argsort(fitness)[:self.population_size // 2]

            # Select the worst particles
            worst_particles = np.argsort(fitness)[-self.population_size // 2:]

            # Create a new population
            new_population = []
            for _ in range(self.population_size):
                # Select a parent from the fittest particles
                parent = random.choice(fittest_particles)

                # Select a child from the worst particles
                child = random.choice(worst_particles)

                # Apply mutation and crossover
                child = self.mutation(child)
                child = self.crossover(parent, child)

                # Add the child to the new population
                new_population.append(child)

            # Evaluate the new population
            fitness = [func(x) for x in new_population]

            # Update the population
            population = new_population

            # Apply simulated annealing
            if _ % 100 == 0:
                T = self.simulated_annealing(T0=self.simulated_annealing_parameters['T0'], Tend=self.simulated_annealing_parameters['Tend'], alpha=self.simulated_annealing_parameters['alpha'])
                for i in range(self.population_size):
                    if random.random() < math.exp((fitness[i] - fitness[random.randint(0, self.population_size - 1)]) / T):
                        population[i] = random.uniform(-5.0, 5.0, self.dim)

        # Return the fittest solution
        return population[np.argmin(fitness)]

    def mutation(self, x):
        # Apply mutation
        x += np.random.normal(0, 0.1, self.dim)
        return x

    def crossover(self, parent, child):
        # Apply crossover
        child = parent[:self.dim // 2] + np.random.uniform(-5.0, 5.0, self.dim // 2)
        return child

    def simulated_annealing(self, T0, Tend, alpha):
        T = T0
        while T > Tend:
            for _ in range(self.population_size):
                if random.random() < math.exp((np.random.uniform(-5.0, 5.0, self.dim) - np.random.uniform(-5.0, 5.0, self.dim)) / T):
                    np.random.uniform(-5.0, 5.0, self.dim)
            T *= alpha
        return T

# Usage
nhea = NHEA(budget=100, dim=10)
nhea('func')