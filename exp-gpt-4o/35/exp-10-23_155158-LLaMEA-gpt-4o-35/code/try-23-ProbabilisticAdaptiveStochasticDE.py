import numpy as np

class ProbabilisticAdaptiveStochasticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8
        self.crossover_prob = 0.75
        self.history_success_rate = []
        self.rescale_budget = max(150, budget // 15)
        
    def adapt_parameters(self):
        if len(self.history_success_rate) >= 5:
            recent_success = np.mean(self.history_success_rate[-5:])
            self.mutation_factor = 0.6 + 0.3 * recent_success  # Adjust mutation factor range
            self.crossover_prob = 0.3 + 0.7 * recent_success  # Adjust crossover probability range

    def hybrid_local_search(self, best_individual):
        step_size = 0.04 + 0.12 * np.random.rand() * np.random.choice([1, -1], p=[0.35, 0.65])
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        candidate = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def dynamic_population_resizing(self, generation):
        if generation % 10 == 0 and self.population_size > 7:  # More frequent resizing
            self.population_size = max(7, int(0.9 * self.population_size))  # Slightly less aggressive reduction

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        generation = 0

        while budget_used < self.budget:
            new_population = []
            new_fitness = []
            self.dynamic_population_resizing(generation)

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
                    self.history_success_rate.append(1)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
                    self.history_success_rate.append(0)

                if budget_used >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            self.adapt_parameters()

            best_idx = np.argmin(fitness)
            local_candidate = self.hybrid_local_search(population[best_idx])
            local_fitness = func(local_candidate)
            budget_used += 1

            if local_fitness < fitness[best_idx]:
                population[best_idx] = local_candidate
                fitness[best_idx] = local_fitness

            generation += 1

            if budget_used >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]