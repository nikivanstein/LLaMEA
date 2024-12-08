import numpy as np

class BeeSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_employed = 20
        self.max_trials = 3
        self.limit = 0.6 * self.num_employed * dim
        self.trials = np.zeros(self.num_employed)
        self.population = np.random.uniform(-5.0, 5.0, (self.num_employed, dim))
        self.fitness = np.array([func(ind) for ind in self.population])
        self.best_solution = self.population[self.fitness.argmin()]
        self.best_fitness = self.fitness.min()

    def employed_bees_phase(self, func):
        new_population = np.copy(self.population)
        for i in range(self.num_employed):
            neighbor = np.random.choice(np.delete(range(self.num_employed), i))
            phi = np.random.uniform(-1, 1, self.dim)
            new_solution = self.population[i] + phi * (self.population[i] - self.population[neighbor])
            new_fitness = func(new_solution)
            if new_fitness < self.fitness[i]:
                new_population[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution
                    self.best_fitness = new_fitness
            else:
                self.trials[i] += 1
        self.population = new_population

    def onlooker_bees_phase(self, func):
        selection_prob = self.fitness.max() - self.fitness
        selection_prob /= selection_prob.sum()
        selected_indices = np.random.choice(self.num_employed, self.num_employed, p=selection_prob)
        
        for i in range(self.num_employed):
            neighbor = np.random.choice(np.delete(selected_indices, i))
            phi = np.random.uniform(-1, 1, self.dim)
            new_solution = self.population[i] + phi * (self.population[i] - self.population[neighbor])
            new_fitness = func(new_solution)
            if new_fitness < self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution
                    self.best_fitness = new_fitness
            else:
                self.trials[i] += 1
    
    def scout_bees_phase(self, func):
        for i in range(self.num_employed):
            if self.trials[i] >= self.limit:
                self.population[i] = np.random.uniform(-5.0, 5.0, self.dim)
                self.fitness[i] = func(self.population[i])
                self.trials[i] = 0

    def __call__(self, func):
        for _ in range(self.budget // (2 * self.num_employed)):
            self.employed_bees_phase(func)
            self.onlooker_bees_phase(func)
            self.scout_bees_phase(func)
        return self.best_solution