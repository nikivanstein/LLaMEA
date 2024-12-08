import numpy as np

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.eval_count = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def quantum_operator(self, population):
        new_population = []
        for i in range(self.population_size):
            quantum_disturbance = np.random.uniform(-1, 1, self.dim) * 0.05
            new_sol = np.clip(population[i] + quantum_disturbance, self.lower_bound, self.upper_bound)
            new_population.append(new_sol)
        return np.array(new_population)

    def mutate(self, population, best):
        indices = np.random.choice(self.population_size, 3, replace=False)
        x1, x2, x3 = population[indices]
        mutant = x1 + self.mutation_factor * (x2 - x3)
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, population, fitness, trial, trial_fitness, idx):
        if trial_fitness < fitness[idx]:
            population[idx] = trial
            fitness[idx] = trial_fitness

    def adapt_population_size(self):
        return max(5, int(self.population_size * (self.budget - self.eval_count) / self.budget))

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)

        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            best = population[best_idx]

            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                mutant = self.mutate(population, best)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1
                self.select(population, fitness, trial, trial_fitness, i)

            if self.eval_count < self.budget:
                population = self.quantum_operator(population)
                fitness = self.evaluate_population(population, func)

            self.population_size = self.adapt_population_size()

        return population[np.argmin(fitness)]