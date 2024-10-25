import numpy as np

class AMQDE_MA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 8 * dim  # Reduced population size
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.memory_F = []
        self.memory_Cr = []
        self.best_solution = None
        self.elite_percentage = 0.2  # Increased elite percentage
        self.num_swarms = 3  # Number of swarms

    def quantum_initialize(self):
        q_population = np.random.rand(self.pop_size, self.dim)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * q_population

    def __call__(self, func):
        populations = [self.quantum_initialize() for _ in range(self.num_swarms)]
        fitnesses = [np.array([func(ind) for ind in pop]) for pop in populations]
        eval_count = self.pop_size * self.num_swarms

        while eval_count < self.budget:
            if len(self.memory_F) > 0 and len(self.memory_Cr) > 0:
                self.F = np.mean(self.memory_F)
                self.Cr = np.mean(self.memory_Cr)

            for swarm_index in range(self.num_swarms):
                population = populations[swarm_index]
                fitness = fitnesses[swarm_index]
                new_population = np.copy(population)
                elite_count = int(self.pop_size * self.elite_percentage)
                elite_indices = np.argsort(fitness)[:elite_count]
                new_population[elite_indices] = population[elite_indices]

                for i in range(self.pop_size):
                    if i in elite_indices:
                        continue

                    x_best = population[np.argmin(fitness)] if self.best_solution is None else self.best_solution
                    indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant = np.clip(x_best + self.F * (x1 - x2 + x2 - x3), self.lower_bound, self.upper_bound)

                    crossover_mask = np.random.rand(self.dim) < self.Cr
                    trial = np.where(crossover_mask, mutant, population[i])

                    trial_fitness = func(trial)
                    eval_count += 1
                    if trial_fitness < fitness[i]:
                        new_population[i] = trial
                        fitness[i] = trial_fitness
                        self.memory_F.append(self.F)
                        self.memory_Cr.append(self.Cr)
                        if len(self.memory_F) > 100:
                            self.memory_F.pop(0)
                        if len(self.memory_Cr) > 100:
                            self.memory_Cr.pop(0)

                        if self.best_solution is None or trial_fitness < func(self.best_solution):
                            self.best_solution = trial

                    if eval_count >= self.budget:
                        break

                populations[swarm_index] = new_population
                fitnesses[swarm_index] = fitness

        best_swarm = np.argmin([np.min(fitness) for fitness in fitnesses])
        best_idx = np.argmin(fitnesses[best_swarm])
        return populations[best_swarm][best_idx], fitnesses[best_swarm][best_idx]