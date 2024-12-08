import numpy as np

class PEAHM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.hypermutation_rate = 0.3

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        return population

    def evaluate(self, func):
        fitnesses = np.array([func(x) for x in self.population])
        min_fitness = np.min(fitnesses)
        max_fitness = np.max(fitnesses)
        if min_fitness == max_fitness:
            return min_fitness
        else:
            return (max_fitness - min_fitness) / (max_fitness - min_fitness + 1e-6)

    def selection(self, fitnesses):
        probabilities = fitnesses / np.sum(fitnesses)
        selection_indices = np.random.choice(self.population_size, size=self.population_size, p=probabilities)
        selected_population = self.population[selection_indices]
        return selected_population

    def crossover(self, parent1, parent2):
        child = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def mutation(self, child):
        for i in range(self.dim):
            if np.random.rand() < self.hypermutation_rate:
                child[i] += np.random.uniform(-1.0, 1.0)
                child[i] = np.clip(child[i], -5.0, 5.0)
        return child

    def hypermutation(self, parent1, parent2):
        child = parent1.copy()
        for i in range(self.dim):
            if np.random.rand() < self.hypermutation_rate:
                child[i] = parent2[i] + np.random.uniform(-1.0, 1.0)
                child[i] = np.clip(child[i], -5.0, 5.0)
        return child

    def evolve(self, func):
        for _ in range(self.budget):
            fitnesses = np.array([func(x) for x in self.population])
            selected_population = self.selection(fitnesses)
            new_population = []
            for i in range(self.population_size):
                parent1 = selected_population[i]
                parent2 = selected_population[(i+1) % self.population_size]
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)
            self.population = np.array(new_population)
        return self.population

    def __call__(self, func):
        self.population = self.evolve(func)
        return self.population

# Example usage:
def func(x):
    return np.sum(x**2)

peahm = PEAHM(budget=100, dim=10)
optimal_solution = peahm(func)
print(optimal_solution)