import numpy as np

class AGPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.crossover_prob = 0.7
        self.mutation_rate = 0.1
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 1.5

    def __call__(self, func):
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.population_size, self.dim) * 0.1
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.population_size

        personal_best = population.copy()
        personal_best_fitness = fitness.copy()

        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        while self.evaluations < self.budget:
            # Update particle velocities and positions
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_component * r1 * (personal_best[i] - population[i])
                    + self.social_component * r2 * (global_best - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate fitness of new position
                current_fitness = func(population[i])
                self.evaluations += 1

                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = current_fitness
                
                # Update global best
                if current_fitness < global_best_fitness:
                    global_best = population[i]
                    global_best_fitness = current_fitness

            # Apply genetic-like crossover and mutation
            if self.evaluations < self.budget:
                for j in range(0, self.population_size, 2):
                    if self.evaluations >= self.budget:
                        break

                    if np.random.rand() < self.crossover_prob:
                        crossover_point = np.random.randint(0, self.dim)
                        population[j, crossover_point:], population[j+1, crossover_point:] = (
                            population[j+1, crossover_point:], population[j, crossover_point:]
                        )

                    if np.random.rand() < self.mutation_rate:
                        mutate_index = np.random.randint(0, self.dim)
                        population[j, mutate_index] = self.lower_bound + np.random.rand() * (self.upper_bound - self.lower_bound)

                    current_fitness_j = func(population[j])
                    current_fitness_j1 = func(population[j+1])
                    self.evaluations += 2

                    if current_fitness_j < personal_best_fitness[j]:
                        personal_best[j] = population[j]
                        personal_best_fitness[j] = current_fitness_j

                    if current_fitness_j1 < personal_best_fitness[j+1]:
                        personal_best[j+1] = population[j+1]
                        personal_best_fitness[j+1] = current_fitness_j1

                    if current_fitness_j < global_best_fitness:
                        global_best = population[j]
                        global_best_fitness = current_fitness_j
                        
                    if current_fitness_j1 < global_best_fitness:
                        global_best = population[j+1]
                        global_best_fitness = current_fitness_j1

        return global_best, global_best_fitness