import numpy as np
import random

class HybridAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.population = self.initialize_population()
        self.best_solution = self.evaluate_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            population.append(solution)
        return population

    def evaluate_population(self):
        best_solution = self.population[0]
        best_score = self.func(best_solution)
        for solution in self.population:
            score = self.func(solution)
            if score < best_score:
                best_solution = solution
                best_score = score
        return best_solution, best_score

    def func(self, x):
        # Replace this with your black box function
        return np.sum(x**2)

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = self.mutate(self.population)
            self.population = self.crossover(self.population)
            self.population = self.evaluate_population()
            self.best_solution, self.best_score = self.population[0]
        return self.best_solution, self.best_score

    def mutate(self, population):
        mutated_population = []
        for _ in range(len(population)):
            solution = population[_]
            mutation_rate = random.random()
            if mutation_rate < 0.1:
                index = random.randint(0, self.dim-1)
                solution[index] += random.uniform(-1, 1)
                if solution[index] < self.lower_bound:
                    solution[index] = self.lower_bound
                elif solution[index] > self.upper_bound:
                    solution[index] = self.upper_bound
            mutated_population.append(solution)
        return mutated_population

    def crossover(self, population):
        crossed_population = []
        for _ in range(len(population)):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = parent1 + parent2
            child = child / np.linalg.norm(child)
            child = child * np.linalg.norm(parent1)
            crossed_population.append(child)
        return crossed_population

    def select(self, population):
        scores = [self.func(solution) for solution in population]
        selected = []
        for _ in range(len(population)):
            score = random.random()
            cumulative_score = 0
            for i, s in enumerate(scores):
                cumulative_score += s
                if score <= cumulative_score:
                    selected.append(population[i])
                    scores.remove(s)
                    break
        return selected

    def hybridize(self, population):
        selected = self.select(population)
        new_population = []
        for _ in range(len(population)):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = parent1 + parent2
            child = child / np.linalg.norm(child)
            child = child * np.linalg.norm(parent1)
            child = child + 0.3 * (parent1 - child)
            child = child + 0.3 * (parent2 - child)
            new_population.append(child)
        return new_population

    def optimize(self, func):
        self.population = self.initialize_population()
        self.best_solution, self.best_score = self.population[0]
        for _ in range(self.budget):
            self.population = self.hybridize(self.population)
            self.population = self.mutate(self.population)
            self.population = self.crossover(self.population)
            self.population = self.evaluate_population()
            self.best_solution, self.best_score = self.population[0]
        return self.best_solution, self.best_score

# Example usage:
budget = 100
dim = 10
algorithm = HybridAlgorithm(budget, dim)
best_solution, best_score = algorithm.optimize(lambda x: x[0]**2 + x[1]**2)
print(f"Best solution: {best_solution}")
print(f"Best score: {best_score}")