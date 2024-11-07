import numpy as np
import random
from pyswarm import pso

class HybridEvoAlg:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.pso_params = {'c1': 0.5, 'c2': 0.5, 'w': 0.8}

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(solution)
        return population

    def fitness(self, solution, func):
        return func(*solution)

    def selection(self, population):
        fitnesses = [self.fitness(solution, func) for solution in population]
        fitnesses = np.array(fitnesses)
        fitness_min = np.min(fitnesses)
        fitness_max = np.max(fitnesses)
        selection_probabilities = (fitnesses - fitness_min) / (fitness_max - fitness_min)
        selected_indices = np.random.choice(len(population), size=self.population_size, p=selection_probabilities)
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        child = [0.5 * (parent1[i] + parent2[i]) for i in range(self.dim)]
        return child

    def mutation(self, solution):
        for i in range(self.dim):
            if random.random() < 0.1:
                solution[i] += random.uniform(-1.0, 1.0)
                solution[i] = max(-5.0, min(5.0, solution[i]))
        return solution

    def local_search(self, solution):
        best_solution = solution
        for i in range(self.dim):
            for new_solution in [solution[:i] + [solution[i] + 1.0] + solution[i+1:],
                                solution[:i] + [solution[i] - 1.0] + solution[i+1:]]:
                fitness = self.fitness(new_solution, func)
                if fitness < self.fitness(best_solution, func):
                    best_solution = new_solution
        return best_solution

    def hybrid_evo_alg(self, func):
        for _ in range(self.budget):
            population = self.selection(self.population)
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)
            self.population = new_population
            best_solution = max(self.population, key=self.fitness)
            if self.fitness(best_solution, func) < self.best_fitness:
                self.best_solution = best_solution
                self.best_fitness = self.fitness(best_solution, func)
            if self.best_fitness < func(0):
                return self.best_solution
        return self.best_solution

    def pso(self, func):
        positions = np.random.uniform(-5.0, 5.0, size=self.dim)
        velocities = np.zeros(self.dim)
        for _ in range(self.budget):
            fitnesses = [func(*position) for position in positions]
            fitness_min = np.min(fitnesses)
            fitness_max = np.max(fitnesses)
            selection_probabilities = (fitnesses - fitness_min) / (fitness_max - fitness_min)
            selected_indices = np.random.choice(len(positions), size=self.population_size, p=selection_probabilities)
            new_positions = [positions[i] for i in selected_indices]
            new_velocities = [velocities[i] for i in selected_indices]
            for i in range(self.dim):
                r1, r2 = random.random(), random.random()
                new_velocities[i] += self.pso_params['c1'] * r1 * (new_positions[i] - positions[i]) + self.pso_params['c2'] * r2 * (new_positions[i] - positions[i])
                new_positions[i] += self.pso_params['w'] * new_velocities[i]
                new_positions[i] = max(-5.0, min(5.0, new_positions[i]))
            positions = new_positions
            velocities = new_velocities
            best_solution = max(positions, key=lambda x: func(*x))
            if func(*best_solution) < self.best_fitness:
                self.best_solution = best_solution
                self.best_fitness = func(*best_solution)
            if self.best_fitness < func(0):
                return self.best_solution
        return self.best_solution

    def hybrid_pso(self, func):
        for _ in range(self.budget):
            population = self.selection(self.population)
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)
            self.population = new_population
            best_solution = max(self.population, key=self.fitness)
            if self.fitness(best_solution, func) < self.best_fitness:
                self.best_solution = best_solution
                self.best_fitness = self.fitness(best_solution, func)
            if self.best_fitness < func(0):
                return self.best_solution
            best_pso_solution = self.pso(func)
            if self.fitness(best_pso_solution, func) < self.best_fitness:
                self.best_solution = best_pso_solution
                self.best_fitness = self.fitness(best_pso_solution, func)
        return self.best_solution

# Test the algorithm
def func(x):
    return sum([i**2 for i in x])

budget = 100
dim = 10
algorithm = HybridEvoAlg(budget, dim)
best_solution = algorithm.hybrid_pso(func)
print(best_solution)