import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_factor = 0.5
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def select_parents(self):
        tournament_size = 5
        indices = np.random.choice(self.population_size, tournament_size, replace=False)
        tournament = self.population[indices]
        tournament_fitness = self.fitness[indices]
        return tournament[np.argsort(tournament_fitness)[:2]]

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < self.crossover_rate
        child = np.where(mask, parent1, parent2)
        return child

    def mutate(self, target, best, fitness_improvement):
        adaptive_factor = 1.0 if fitness_improvement > 0 else 0.5
        r1, r2, r3 = self.population[np.random.choice(self.population_size, 3, replace=False)]
        mutant = target + adaptive_factor * self.mutation_factor * (best - target) + self.mutation_factor * (r1 - r2) + self.mutation_factor * (r2 - r3)
        return np.clip(mutant, self.lb, self.ub)

    def __call__(self, func):
        num_evaluations = 0
        self.evaluate_population(func)
        num_evaluations += self.population_size

        while num_evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            best_idx = np.argmin(self.fitness)
            best = self.population[best_idx]

            for i in range(self.population_size):
                target = self.population[i]
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                previous_fitness = self.fitness[i]
                mutant = self.mutate(target, best, self.fitness[best_idx] - previous_fitness)

                if np.random.rand() < self.crossover_rate:
                    trial_vector = mutant
                else:
                    trial_vector = child

                trial_fitness = func(trial_vector)
                num_evaluations += 1

                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial_vector
                    self.fitness[i] = trial_fitness
                else:
                    new_population[i] = target

                if num_evaluations >= self.budget:
                    break

            self.population = new_population

        return self.population[np.argmin(self.fitness)]