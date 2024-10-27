import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class HSBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01
        self.inheritance_rate = 0.1
        self.boundaries = [(-5.0, 5.0), (-1.0, 1.0)]

    def __call__(self, func):
        population = self.initialize_population()
        for _ in range(self.budget):
            new_population = self.selection(population)
            new_population = self.crossover(new_population)
            new_population = self.mutation(new_population)
            population = self.inheritance(population, new_population)
        return self.evaluate_fitness(population)

    def initialize_population(self):
        return [np.random.uniform(self.boundaries[0][0], self.boundaries[0][1], self.dim) for _ in range(self.population_size)]

    def selection(self, population):
        fitnesses = [self.evaluate_fitness(individual) for individual in population]
        selected_indices = np.argsort(fitnesses)[-self.population_size:]
        selected_individuals = [population[i] for i in selected_indices]
        return selected_individuals

    def crossover(self, population):
        children = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                parent1, parent2 = population[i], population[i + 1]
                child = (parent1 + parent2) / 2
                children.append(child)
            else:
                child = population[i]
                children.append(child)
        return children

    def mutation(self, population):
        mutated_population = [individual + np.random.normal(0, 0.1, self.dim) for individual in population]
        return mutated_population

    def inheritance(self, population, new_population):
        child = np.random.choice(new_population, self.population_size, replace=False)
        child = np.concatenate((child[:self.population_size // 2], new_population[self.population_size // 2:]))
        return child

    def evaluate_fitness(self, population):
        fitnesses = [self.evaluate_fitness(individual) for individual in population]
        return np.mean(fitnesses)

def evaluateBBOB(func, population, budget):
    hsbbo = HSBBO(budget, 10)
    return hsbbo(population)

# Example usage:
func = lambda x: x**2
population = evaluateBBOB(func, [np.linspace(-10, 10, 100)] * 10, 100)
print(population)
print(evaluateBBOB(func, population, 100))