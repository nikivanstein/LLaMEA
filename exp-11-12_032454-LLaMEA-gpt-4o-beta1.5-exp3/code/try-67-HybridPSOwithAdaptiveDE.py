import numpy as np

class HybridPSOwithAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w = 0.7   # inertia weight
        self.F = 0.5   # differential weight
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break

                # Particle Swarm Optimization update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - population[i]) +
                                 self.c2 * r2 * (global_best - population[i]))
                population[i] = np.clip(population[i] + velocities[i], self.lb, self.ub)

                # Evaluate new position
                fitness[i] = func(population[i])
                num_evaluations += 1

                # Update personal best
                if fitness[i] < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness[i]

                # Update global best
                if fitness[i] < global_best_fitness:
                    global_best = population[i]
                    global_best_fitness = fitness[i]

                # Adaptive Differential Evolution
                if num_evaluations < self.budget:
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant_vector = personal_best[a] + self.F * (personal_best[b] - personal_best[c])
                    mutant_vector = np.clip(mutant_vector, self.lb, self.ub)

                    crossover_mask = np.random.rand(self.dim) < self.CR
                    trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                    
                    trial_fitness = func(trial_vector)
                    num_evaluations += 1

                    if trial_fitness < fitness[i]:
                        population[i] = trial_vector
                        fitness[i] = trial_fitness
                        if trial_fitness < personal_best_fitness[i]:
                            personal_best[i] = trial_vector
                            personal_best_fitness[i] = trial_fitness
                        if trial_fitness < global_best_fitness:
                            global_best = trial_vector
                            global_best_fitness = trial_fitness

        return global_best, global_best_fitness