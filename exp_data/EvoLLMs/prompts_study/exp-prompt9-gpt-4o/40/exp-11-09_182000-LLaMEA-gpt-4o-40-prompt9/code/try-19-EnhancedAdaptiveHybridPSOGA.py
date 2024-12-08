import numpy as np

class EnhancedAdaptiveHybridPSOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.c1 = 1.4  # Slightly adjusted cognitive parameter for exploration
        self.c2 = 2.6  # Slightly adjusted social parameter for stronger convergence
        self.w = 0.5  # Adaptive inertia weight for dynamic balance
        self.pa = 0.25
        self.pm = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pheromone_decay = 0.9  # New parameter for pheromone decay
        self.pheromones = np.ones((self.pop_size, self.dim))  # Initialize pheromones

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]

        evaluations = self.pop_size

        while evaluations < self.budget:
            # PSO Update with pheromone influence
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - population[i]) +
                                 self.c2 * r2 * (global_best - population[i]) +
                                 0.1 * self.pheromones[i])  # Pheromone influence
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                f_val = func(population[i])
                evaluations += 1

                # Update personal and global bests
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = population[i].copy()

                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                self.pheromones[i] = self.pheromone_decay * self.pheromones[i]  # Decay pheromones

                if evaluations >= self.budget:
                    break

            # Genetic Algorithm-inspired crossover and mutation
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Crossover
                mate_idx = np.random.randint(0, self.pop_size)
                offspring = 0.5 * (population[i] + population[mate_idx])
                offspring = np.clip(offspring, self.lower_bound, self.upper_bound)

                # Mutation
                if np.random.rand() < self.pm:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    offspring += mutation_vector
                    offspring = np.clip(offspring, self.lower_bound, self.upper_bound)

                f_offspring = func(offspring)
                evaluations += 1

                if f_offspring < personal_best_values[i]:
                    personal_best_values[i] = f_offspring
                    personal_best[i] = offspring.copy()

                    if f_offspring < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                self.pheromones[i] = (1 - self.pheromone_decay) + self.pheromone_decay * self.pheromones[i]  # Update pheromones

        return global_best