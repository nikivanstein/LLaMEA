import numpy as np
import random

class SwarmOfSpheres:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.sphere_radius = 1.0
        self.learning_rate = 0.1
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        if self.budget == 0:
            return self.best_solution

        for _ in range(self.budget):
            # Initialize population with random solutions
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

            # Evaluate population fitness
            fitness = [func(x) for x in population]

            # Update best solution if necessary
            if min(fitness) < self.best_fitness:
                self.best_solution = population[np.argmin(fitness)]
                self.best_fitness = min(fitness)

            # Apply swarm of spheres evolution
            for _ in range(10):
                # Select parents using tournament selection
                parents = np.array([population[np.random.choice(range(self.population_size)), :] for _ in range(2)])

                # Apply crossover and mutation
                offspring = []
                for _ in range(self.population_size):
                    parent1, parent2 = parents[np.random.choice(range(2))]
                    child = (parent1 + parent2) * 0.5 + np.random.uniform(-self.sphere_radius, self.sphere_radius, self.dim)
                    child = np.clip(child, -5.0, 5.0)
                    offspring.append(child)

                # Update population
                population = np.array(offspring)

            # Update learning rate
            self.learning_rate *= 0.9

            # Refine individual solutions with probability 0.3
            refine_prob = 0.3
            refined_population = population.copy()
            for i in range(self.population_size):
                if random.random() < refine_prob:
                    # Refine solution by changing one dimension randomly
                    refined_solution = refined_population[i].copy()
                    dimension = np.random.choice(self.dim)
                    refined_solution[dimension] += np.random.uniform(-1.0, 1.0)
                    refined_solution = np.clip(refined_solution, -5.0, 5.0)
                    refined_population[i] = refined_solution

            # Update population with refined solutions
            population = refined_population

        return self.best_solution

# Example usage:
def func(x):
    return sum(x**2)

swarm = SwarmOfSpheres(budget=100, dim=10)
best_solution = swarm(func)
print(best_solution)