import numpy as np

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.6
        self.crossover_prob = 0.8
        self.success_rate_history = []
        self.rescale_interval = budget // 8

    def adapt_parameters(self):
        if len(self.success_rate_history) >= 5:
            recent_success = np.mean(self.success_rate_history[-5:])
            self.mutation_factor = 0.5 + 0.4 * recent_success  # more dynamic mutation
            self.crossover_prob = 0.7 + 0.2 * recent_success  # adjust crossover

    def intensification(self, best_individual):
        step_size = 0.04 + 0.02 * np.random.rand() * np.random.choice([1, -1])
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        candidate = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def dynamic_population_control(self, generation):
        if generation % 8 == 0 and self.population_size > 20:
            self.population_size = max(20, int(0.8 * self.population_size))

    def elite_preservation(self, population, fitness):
        elite_idx = np.argmin(fitness)
        return population[elite_idx], fitness[elite_idx]

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        generation = 0

        while budget_used < self.budget:
            new_population = []
            new_fitness = []
            self.dynamic_population_control(generation)

            for i in range(self.population_size):
                idxs = np.delete(np.arange(self.population_size), i)
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
                    self.success_rate_history.append(1)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
                    self.success_rate_history.append(0)

                if budget_used >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            self.adapt_parameters()

            best_individual, best_fitness = self.elite_preservation(population, fitness)

            if budget_used < self.budget and np.random.rand() < 0.5:
                intense_candidate = self.intensification(best_individual)
                intense_fitness = func(intense_candidate)
                budget_used += 1

                if intense_fitness < best_fitness:
                    best_individual = intense_candidate
                    best_fitness = intense_fitness

            generation += 1

            if budget_used >= self.budget:
                break

        return best_individual, best_fitness