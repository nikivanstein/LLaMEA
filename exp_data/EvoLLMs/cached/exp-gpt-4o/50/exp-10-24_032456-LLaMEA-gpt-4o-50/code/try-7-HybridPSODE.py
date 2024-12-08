import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(0.5 * dim)
        self.current_evaluations = 0
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.F = 0.8  # Differential weight
        self.prob_crossover = 0.7  # Crossover probability

    def __call__(self, func):
        # Initialize PSO
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.random.uniform(
            -1, 1, (self.population_size, self.dim)
        )
        
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        best_idx = np.argmin(fitness)
        global_best_position = population[best_idx]
        global_best_fitness = fitness[best_idx]

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                if self.current_evaluations >= self.budget:
                    break

                # Update velocities and positions
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - population[i])
                    + self.c2 * r2 * (global_best_position - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = func(population[i])
                self.current_evaluations += 1

                # Update personal and global bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = new_fitness

                if new_fitness < global_best_fitness:
                    global_best_position = population[i]
                    global_best_fitness = new_fitness

                # Hybrid with Differential Evolution
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.prob_crossover
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                self.current_evaluations += 1
                if trial_fitness < new_fitness:
                    population[i] = trial
                    new_fitness = trial_fitness

                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = new_fitness

                if new_fitness < global_best_fitness:
                    global_best_position = population[i]
                    global_best_fitness = new_fitness

        return global_best_position