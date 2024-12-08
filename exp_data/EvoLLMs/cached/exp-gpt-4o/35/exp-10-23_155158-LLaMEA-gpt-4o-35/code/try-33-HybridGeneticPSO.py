import numpy as np

class HybridGeneticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.mutation_rate = 0.2
        self.cross_probability = 0.9

    def crossover(self, parent1, parent2):
        cross_points = np.random.rand(self.dim) < self.cross_probability
        offspring = np.where(cross_points, parent1, parent2)
        return offspring

    def mutate(self, individual):
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate
        random_changes = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        individual = np.where(mutation_mask, random_changes, individual)
        return np.clip(individual, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)
        budget_used = self.population_size

        while budget_used < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best[i] - population[i]) +
                                 self.social_const * r2 * (global_best - population[i]))
                candidate = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                budget_used += 1

                if candidate_fitness < personal_best_fitness[i]:
                    personal_best[i] = candidate
                    personal_best_fitness[i] = candidate_fitness

                    if candidate_fitness < global_best_fitness:
                        global_best = candidate
                        global_best_fitness = candidate_fitness

            # Apply genetic operations
            new_population = []
            for _ in range(self.population_size):
                parents = np.random.choice(self.population_size, 2, replace=False)
                offspring = self.crossover(personal_best[parents[0]], personal_best[parents[1]])
                offspring = self.mutate(offspring)
                new_population.append(offspring)

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])
            budget_used += self.population_size

            # Update personal and global best
            for i in range(self.population_size):
                if fitness[i] < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness[i]

                    if fitness[i] < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness[i]

        return global_best, global_best_fitness