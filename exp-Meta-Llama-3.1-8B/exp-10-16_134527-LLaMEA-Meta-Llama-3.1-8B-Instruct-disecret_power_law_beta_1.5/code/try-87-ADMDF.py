import numpy as np
import random

class ADMDF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.levy_flight_probability = 0.05
        self.self_adaptive_parameter_control_probability = 0.05
        self.sigma = 0.1
        self.F = 0.5
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
        self.f_best = float('inf')
        self.probability_change_individual_lines = 0.9

    def levy_flight(self, x):
        sigma = 0.01
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, sigma)
        step_size = np.abs(u) / np.abs(v)
        return x + step_size * (x - np.random.uniform(-5.0, 5.0, self.dim))

    def self_adaptive_parameter_control(self):
        self.sigma = np.random.uniform(0.01, 0.1)
        self.F = np.random.uniform(0.1, 1.0)

    def crossover(self, x1, x2):
        if random.random() < self.crossover_probability:
            return (x1 + x2) / 2
        else:
            return x1

    def mutation(self, x):
        if random.random() < self.mutation_probability:
            return self.levy_flight(x)
        else:
            return x + np.random.uniform(-1.0, 1.0, self.dim)

    def update_selected_solution(self, x):
        x_new = self.mutation(x)
        if random.random() < self.probability_change_individual_lines:
            index = random.randint(0, self.dim - 1)
            x_new[index] = np.random.uniform(-5.0, 5.0)
        else:
            x_new = self.crossover(x, self.levy_flight(x))
        return x_new

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                x_new = self.update_selected_solution(population[i])
                if func(x_new) < func(population[i]):
                    population[i] = x_new
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            if random.random() < self.self_adaptive_parameter_control_probability:
                self.self_adaptive_parameter_control()
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.levy_flight_probability = 0.05
        self.self_adaptive_parameter_control_probability = 0.05
        self.sigma = 0.1
        self.F = 0.5
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
        self.f_best = float('inf')
        self.probability_change_individual_lines = 0.9

    def levy_flight(self, x):
        sigma = 0.01
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, sigma)
        step_size = np.abs(u) / np.abs(v)
        return x + step_size * (x - np.random.uniform(-5.0, 5.0, self.dim))

    def self_adaptive_parameter_control(self):
        self.sigma = np.random.uniform(0.01, 0.1)
        self.F = np.random.uniform(0.1, 1.0)

    def crossover(self, x1, x2):
        if random.random() < self.crossover_probability:
            return (x1 + x2) / 2
        else:
            return x1

    def mutation(self, x):
        if random.random() < self.mutation_probability:
            return self.levy_flight(x)
        else:
            return x + np.random.uniform(-1.0, 1.0, self.dim)

    def update_selected_solution(self, x):
        x_new = self.mutation(x)
        if random.random() < self.probability_change_individual_lines:
            index = random.randint(0, self.dim - 1)
            x_new[index] = np.random.uniform(-5.0, 5.0)
        else:
            x_new = self.crossover(x, self.levy_flight(x))
        return x_new

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                x_new = self.update_selected_solution(population[i])
                if func(x_new) < func(population[i]):
                    population[i] = x_new
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            if random.random() < self.self_adaptive_parameter_control_probability:
                self.self_adaptive_parameter_control()
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)

    def novel_heuristic(self, x):
        x_new = self.mutation(x)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        return x_new

    def novel_heuristic2(self, x):
        x_new = self.mutation(x)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        return x_new

    def novel_heuristic3(self, x):
        x_new = self.mutation(x)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        return x_new

    def novel_heuristic4(self, x):
        x_new = self.mutation(x)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        return x_new

    def novel_heuristic5(self, x):
        x_new = self.mutation(x)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        return x_new

    def novel_heuristic6(self, x):
        x_new = self.mutation(x)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        return x_new

    def novel_heuristic7(self, x):
        x_new = self.mutation(x)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        return x_new

    def novel_heuristic8(self, x):
        x_new = self.mutation(x)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        return x_new

    def novel_heuristic9(self, x):
        x_new = self.mutation(x)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        return x_new

    def novel_heuristic10(self, x):
        x_new = self.mutation(x)
        index1 = random.randint(0, self.dim - 1)
        index2 = random.randint(0, self.dim - 1)
        if index1!= index2:
            x_new[index1], x_new[index2] = x_new[index2], x_new[index1]
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        index = random.randint(0, self.dim - 1)
        x_new[index] = np.random.uniform(-5.0, 5.0)
        return x_new

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                x_new = self.novel_heuristic10(population[i])
                if func(x_new) < func(population[i]):
                    population[i] = x_new
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            if random.random() < self.self_adaptive_parameter_control_probability:
                self.self_adaptive_parameter_control()
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)

# Description: Novel Metaheuristic Algorithm with 10 novel heuristics to solve black box optimization problems.
# Code: