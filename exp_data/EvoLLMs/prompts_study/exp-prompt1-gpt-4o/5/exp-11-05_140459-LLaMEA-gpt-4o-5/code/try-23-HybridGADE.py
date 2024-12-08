import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.base_crossover_rate = 0.7
        self.mutation_factor = 0.5
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def select_parents(self):
        indices = np.arange(self.population_size)
        np.random.shuffle(indices)
        return self.population[indices[:2]]

    def crossover(self, parent1, parent2, dynamic_crossover_rate):
        mask = np.random.rand(self.dim) < dynamic_crossover_rate
        child = np.where(mask, parent1, parent2)
        return child

    def mutate(self, target, best):
        r1, r2, r3 = self.population[np.random.choice(self.population_size, 3, replace=False)]
        mutant = target + self.mutation_factor * (best - target) + self.mutation_factor * (r1 - r2) + self.mutation_factor * (r2 - r3)
        return np.clip(mutant, self.lb, self.ub)

    def __call__(self, func):
        num_evaluations = 0
        self.evaluate_population(func)
        num_evaluations += self.population_size

        while num_evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            best_idx = np.argmin(self.fitness)
            best = self.population[best_idx]
            
            # Calculate diversity for adaptive crossover rate
            diversity = np.mean(np.std(self.population, axis=0))
            dynamic_crossover_rate = self.base_crossover_rate + (0.3 * (1 - diversity))

            for i in range(self.population_size):
                target = self.population[i]
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2, dynamic_crossover_rate)
                mutant = self.mutate(target, best)

                if np.random.rand() < dynamic_crossover_rate:
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