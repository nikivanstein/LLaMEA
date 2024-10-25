import numpy as np

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Increased initial population size for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8  # Slightly higher mutation factor for exploration
        self.crossover_prob = 0.85  # Adjusted crossover probability for better exploration-exploitation balance
        self.success_rate_history = []
        self.rescale_interval = max(120, budget // 8)  # Adapted rescale interval for better parameter adaptation

    def adapt_parameters(self):
        if len(self.success_rate_history) >= 5:
            recent_success = np.mean(self.success_rate_history[-5:])
            self.mutation_factor = 0.5 + 0.4 * recent_success  # Adjusted mutation factor range
            self.crossover_prob = 0.7 + 0.2 * recent_success  # Adjusted crossover probability range

    def intensification(self, best_individual):
        step_size = 0.07 + 0.02 * np.random.rand() * np.random.choice([1, -1])
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        candidate = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def diversification(self, population):
        perturb = np.random.normal(0, 0.1, population.shape)
        diversified_population = np.clip(population + perturb, self.lower_bound, self.upper_bound)
        return diversified_population

    def dynamic_population_control(self, generation):
        if generation % 8 == 0 and self.population_size > 20:
            self.population_size = max(20, int(0.8 * self.population_size))  # Adjusted population control strategy

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

            population = self.diversification(np.array(new_population))  # Added diversification step
            fitness = np.array(new_fitness)
            self.adapt_parameters()

            best_idx = np.argmin(fitness)
            if budget_used < self.budget and np.random.rand() < 0.45:  # Adjusted compromise probability for intensification
                intense_candidate = self.intensification(population[best_idx])
                intense_fitness = func(intense_candidate)
                budget_used += 1

                if intense_fitness < fitness[best_idx]:
                    population[best_idx] = intense_candidate
                    fitness[best_idx] = intense_fitness

            generation += 1

            if budget_used >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]