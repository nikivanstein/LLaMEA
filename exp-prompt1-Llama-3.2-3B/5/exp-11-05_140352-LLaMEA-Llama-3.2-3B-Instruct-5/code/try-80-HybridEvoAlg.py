import numpy as np
import random

class HybridEvoAlg:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.chaos_prob = 0.1

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
                if random.random() < self.chaos_prob:
                    child = self.chaos_optimization(child)
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

    def chaos_optimization(self, solution):
        # Apply chaos optimization by adding a small random value to each dimension
        for i in range(self.dim):
            solution[i] += random.uniform(-0.1, 0.1)
            solution[i] = max(-5.0, min(5.0, solution[i]))
        return solution

# Example usage:
import numpy as np
import random
from functools import partial

# Define the BBOB test suite functions
def func1(x):
    return x[0]**2 + x[1]**2

def func2(x):
    return x[0]**3 - 3*x[0]**2*x[1] + x[1]**3

def func3(x):
    return x[0]**4 - 4*x[0]**3*x[1] + 6*x[0]**2*x[1]**2 - x[1]**4

def func4(x):
    return x[0]**5 - 5*x[0]**4*x[1] + 10*x[0]**3*x[1]**2 - 5*x[0]**2*x[1]**3 + x[1]**5

def func5(x):
    return x[0]**6 - 6*x[0]**5*x[1] + 15*x[0]**4*x[1]**2 - 10*x[0]**3*x[1]**3 + 5*x[0]**2*x[1]**4 - x[1]**6

def func6(x):
    return x[0]**7 - 7*x[0]**6*x[1] + 21*x[0]**5*x[1]**2 - 35*x[0]**4*x[1]**3 + 21*x[0]**3*x[1]**4 - 7*x[0]**2*x[1]**5 + x[1]**7

def func7(x):
    return x[0]**8 - 8*x[0]**7*x[1] + 56*x[0]**6*x[1]**2 - 112*x[0]**5*x[1]**3 + 56*x[0]**4*x[1]**4 - 16*x[0]**3*x[1]**5 + 16*x[0]**2*x[1]**6 - x[1]**8

def func8(x):
    return x[0]**9 - 9*x[0]**8*x[1] + 84*x[0]**7*x[1]**2 - 126*x[0]**6*x[1]**3 + 84*x[0]**5*x[1]**4 - 24*x[0]**4*x[1]**5 + 16*x[0]**3*x[1]**6 - 6*x[0]**2*x[1]**7 + x[1]**9

def func9(x):
    return x[0]**10 - 10*x[0]**9*x[1] + 120*x[0]**8*x[1]**2 - 210*x[0]**7*x[1]**3 + 252*x[0]**6*x[1]**4 - 120*x[0]**5*x[1]**5 + 30*x[0]**4*x[1]**6 - 16*x[0]**3*x[1]**7 + 6*x[0]**2*x[1]**8 - x[1]**10

def func10(x):
    return x[0]**11 - 11*x[0]**10*x[1] + 330*x[0]**9*x[1]**2 - 462*x[0]**8*x[1]**3 + 462*x[0]**7*x[1]**4 - 165*x[0]**6*x[1]**5 + 33*x[0]**5*x[1]**6 - 16*x[0]**4*x[1]**7 + 6*x[0]**3*x[1]**8 - x[1]**11

def func11(x):
    return x[0]**12 - 12*x[0]**11*x[1] + 495*x[0]**10*x[1]**2 - 792*x[0]**9*x[1]**3 + 792*x[0]**8*x[1]**4 - 330*x[0]**7*x[1]**5 + 66*x[0]**6*x[1]**6 - 16*x[0]**5*x[1]**7 + 6*x[0]**4*x[1]**8 - x[1]**12

def func12(x):
    return x[0]**13 - 13*x[0]**12*x[1] + 715*x[0]**11*x[1]**2 - 1260*x[0]**10*x[1]**3 + 1540*x[0]**9*x[1]**4 - 1260*x[0]**8*x[1]**5 + 462*x[0]**7*x[1]**6 - 114*x[0]**6*x[1]**7 + 16*x[0]**5*x[1]**8 - x[1]**13

def func13(x):
    return x[0]**14 - 14*x[0]**13*x[1] + 1716*x[0]**12*x[1]**2 - 3003*x[0]**11*x[1]**3 + 4620*x[0]**10*x[1]**4 - 5040*x[0]**9*x[1]**5 + 2640*x[0]**8*x[1]**6 - 720*x[0]**7*x[1]**7 + 80*x[0]**6*x[1]**8 - 16*x[0]**5*x[1]**9 + x[1]**14

def func14(x):
    return x[0]**15 - 15*x[0]**14*x[1] + 2275*x[0]**13*x[1]**2 - 4500*x[0]**12*x[1]**3 + 6750*x[0]**11*x[1]**4 - 9000*x[0]**10*x[1]**5 + 7200*x[0]**9*x[1]**6 - 3000*x[0]**8*x[1]**7 + 720*x[0]**7*x[1]**8 - 80*x[0]**6*x[1]**9 + 16*x[0]**5*x[1]**10 - x[1]**15

def func15(x):
    return x[0]**16 - 16*x[0]**15*x[1] + 3180*x[0]**14*x[1]**2 - 6720*x[0]**13*x[1]**3 + 12000*x[0]**12*x[1]**4 - 21600*x[0]**11*x[1]**5 + 21600*x[0]**10*x[1]**6 - 9000*x[0]**9*x[1]**7 + 2160*x[0]**8*x[1]**8 - 240*x[0]**7*x[1]**9 + 16*x[0]**6*x[1]**10 - x[1]**16

# Evaluate the function on the BBOB test suite
def evaluateBBOB(func):
    results = []
    for f in [func1, func2, func3, func4, func5, func6, func7, func8, func9, func10, func11, func12, func13, func14, func15]:
        best_solution = HybridEvoAlg(50, 2).hybrid_evo_alg(f)
        results.append((f.__name__, best_solution, f(best_solution)))
    return results

# Evaluate the BBOB test suite
results = evaluateBBOB(func1)
for result in results:
    print(result)

# Example usage with a different function
# results = evaluateBBOB(func2)
# for result in results:
#     print(result)