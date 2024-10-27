import numpy as np
import random
import math

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_solution = self.get_best_solution()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.generate_individual()
            population.append(individual)
        return population

    def generate_individual(self):
        individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
        return individual

    def get_best_solution(self):
        best_solution = self.population[0]
        best_score = self.evaluate(best_solution)
        return best_solution, best_score

    def evaluate(self, solution):
        score = 0
        for i in range(self.dim):
            score += math.sin(solution[i])
        return score

    def simulate_annealing(self, solution):
        temperature = 1000
        cooling_rate = 0.99
        for _ in range(self.budget):
            if random.random() < 0.3:
                # Permutation mutation
                i, j = random.sample(range(self.dim), 2)
                solution[i], solution[j] = solution[j], solution[i]
            else:
                # Genetic mutation
                if random.random() < 0.3:
                    i = random.randint(0, self.dim - 1)
                    solution[i] += random.uniform(-1.0, 1.0)
                    solution[i] = max(-5.0, min(5.0, solution[i]))
            score = self.evaluate(solution)
            if score > self.evaluate(self.best_solution):
                self.best_solution = solution
            if temperature > 1:
                probability = math.exp((score - self.evaluate(self.best_solution)) / temperature)
                if random.random() < probability:
                    self.best_solution = solution
            temperature *= cooling_rate

    def optimize(self, func):
        for _ in range(self.budget):
            solution, score = self.get_best_solution()
            self.simulate_annealing(solution)
        return self.best_solution

# Example usage:
budget = 100
dim = 10
hybrid = HybridOptimization(budget, dim)
best_solution = hybrid.optimize(lambda x: x[0]**2 + x[1]**2)
print(best_solution)