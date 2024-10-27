import numpy as np
import random

class EADS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_values = [self.evaluate_func(func) for func in self.population]
        self.best_solution = self.get_best_solution()
        self.budget_evaluations = 0

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
            population.append(individual)
        return population

    def evaluate_func(self, func):
        if self.budget_evaluations < self.budget:
            self.budget_evaluations += 1
            return func(*individual)
        else:
            return float('inf')

    def get_best_solution(self):
        best_solution = self.population[np.argmin(self.fitness_values)]
        return best_solution

    def update_solution(self, solution):
        if self.budget_evaluations < self.budget:
            new_solution = solution.copy()
            for _ in range(int(self.budget_evaluations * 0.3)):
                i = random.randint(0, self.dim - 1)
                new_solution[i] += random.uniform(-1, 1)
                new_solution[i] = max(self.lower_bound, min(new_solution[i], self.upper_bound))
            self.fitness_values = [self.evaluate_func(func) for func in self.population]
            self.best_solution = self.get_best_solution()
            self.population = [new_solution]
            self.fitness_values = [self.evaluate_func(func) for func in self.population]
            return self.best_solution
        else:
            return solution

# Example usage
def func1(x):
    return sum(x**2)

def func2(x):
    return sum(x**3)

def func3(x):
    return sum(x**4)

def func4(x):
    return sum(x**5)

def func5(x):
    return sum(x**6)

def func6(x):
    return sum(x**7)

def func7(x):
    return sum(x**8)

def func8(x):
    return sum(x**9)

def func9(x):
    return sum(x**10)

def func10(x):
    return sum(x**11)

def func11(x):
    return sum(x**12)

def func12(x):
    return sum(x**13)

def func13(x):
    return sum(x**14)

def func14(x):
    return sum(x**15)

def func15(x):
    return sum(x**16)

def func16(x):
    return sum(x**17)

def func17(x):
    return sum(x**18)

def func18(x):
    return sum(x**19)

def func19(x):
    return sum(x**20)

def func20(x):
    return sum(x**21)

def func21(x):
    return sum(x**22)

def func22(x):
    return sum(x**23)

def func23(x):
    return sum(x**24)

def func24(x):
    return sum(x**25)

eads = EADS(budget=100, dim=10)
for _ in range(100):
    eads.update_solution(eads.best_solution)
    print(eads.best_solution)