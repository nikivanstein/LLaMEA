import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.history_success_rate = []
        self.inertia_weight = 0.5
        self.personal_best_positions = None
        self.global_best_position = None

    def scale_parameters(self):
        if len(self.history_success_rate) >= 10:
            avg_success = np.mean(self.history_success_rate[-10:])
            self.mutation_factor = 0.5 + 0.4 * avg_success
            self.crossover_prob = 0.5 + 0.4 * avg_success
            self.inertia_weight = 0.4 + 0.3 * avg_success

    def local_search(self, best_individual):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        candidate = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def initialize_pso(self, population):
        self.personal_best_positions = np.copy(population)
        self.global_best_position = population[np.argmin([func(ind) for ind in population])]

    def update_pso(self, population, velocities, personal_best_values):
        for i, individual in enumerate(population):
            r1, r2 = np.random.rand(2)
            velocities[i] = (self.inertia_weight * velocities[i] +
                             r1 * (self.personal_best_positions[i] - individual) +
                             r2 * (self.global_best_position - individual))
            population[i] = np.clip(individual + velocities[i], self.lower_bound, self.upper_bound)
            fitness_value = func(individual)
            if fitness_value < personal_best_values[i]:
                self.personal_best_positions[i] = individual
                personal_best_values[i] = fitness_value
                if fitness_value < func(self.global_best_position):
                    self.global_best_position = individual

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        personal_best_values = np.copy(fitness)
        budget_used = self.population_size

        self.initialize_pso(population)

        while budget_used < self.budget:
            new_population = []
            new_fitness = []

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, population[i])
                trial_fitness = func(trial_vector)
                budget_used += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    new_fitness.append(trial_fitness)
                    self.history_success_rate.append(1)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
                    self.history_success_rate.append(0)

                if budget_used >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            self.scale_parameters()

            # PSO update
            self.update_pso(population, velocities, personal_best_values)

            best_idx = np.argmin(fitness)
            local_candidate = self.local_search(population[best_idx])
            local_fitness = func(local_candidate)
            budget_used += 1

            if local_fitness < fitness[best_idx]:
                population[best_idx] = local_candidate
                fitness[best_idx] = local_fitness

            if budget_used >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]