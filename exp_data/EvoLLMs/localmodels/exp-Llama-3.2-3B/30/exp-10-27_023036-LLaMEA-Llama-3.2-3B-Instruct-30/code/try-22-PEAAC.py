import numpy as np

class PEAAC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.adaptive_crossover_rate = 0.3

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def fitness(self, func, individual):
        return func(individual)

    def selection(self, population, func):
        fitnesses = [self.fitness(func, individual) for individual in population]
        fitnesses = np.array(fitnesses)
        probabilities = np.exp(-fitnesses) / np.sum(np.exp(-fitnesses))
        selected_indices = np.random.choice(len(population), size=self.population_size, p=probabilities)
        selected_individuals = [population[i] for i in selected_indices]
        return selected_individuals

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            child = np.random.uniform(-5.0, 5.0, self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            return child
        else:
            return parent1

    def mutation(self, individual):
        if np.random.rand() < self.mutation_rate:
            index = np.random.randint(0, self.dim)
            individual[index] = np.random.uniform(-5.0, 5.0)
        return individual

    def update(self, func):
        self.population = self.selection(self.population, func)
        new_population = []
        for i in range(self.population_size):
            parent1 = self.population[i]
            parent2 = self.population[(i+1) % self.population_size]
            child = self.crossover(parent1, parent2)
            child = self.mutation(child)
            new_population.append(child)
        self.population = new_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.update(func)
            best_individual = max(self.population, key=lambda x: self.fitness(func, x))
            print(f'Iteration {_+1}: Best individual = {best_individual}, Fitness = {self.fitness(func, best_individual)}')