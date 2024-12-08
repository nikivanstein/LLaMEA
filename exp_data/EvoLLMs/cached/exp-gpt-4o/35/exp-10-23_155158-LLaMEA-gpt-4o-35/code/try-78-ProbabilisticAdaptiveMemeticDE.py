import numpy as np

class ProbabilisticAdaptiveMemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Initial population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8  # Base mutation factor
        self.crossover_prob = 0.85  # Base crossover probability
        self.success_rate_history = []
        self.memory_size = 12
        self.memory = np.zeros((self.memory_size, 2))
        self.learning_rate = 0.1  # New adaptive learning rate for mutation factor

    def adapt_parameters(self):
        if len(self.success_rate_history) >= self.memory_size:
            recent_success = np.mean(self.success_rate_history[-self.memory_size:])
            base_factor = 0.65 + 0.35 * recent_success
            self.mutation_factor = np.clip(base_factor + self.learning_rate * np.random.randn(), 0.5, 1.2)
            self.crossover_prob = np.clip(0.85 + 0.15 * recent_success + 0.05 * np.random.randn(), 0.6, 1.0)
            self.memory = np.roll(self.memory, -1, axis=0)
            self.memory[-1] = [self.mutation_factor, self.crossover_prob]

    def intensification(self, best_individual):
        step_size = 0.02 + 0.03 * np.random.randn() * np.random.choice([1, -1])
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        candidate = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def dynamic_population_control(self, generation):
        if generation % 4 == 0 and self.population_size > 15:
            self.population_size = max(15, int(0.65 * self.population_size))

    def preserve_diversity(self, population):
        diversity_threshold = 0.1
        if np.std(population) < diversity_threshold:
            # Increase diversity by adding noise to some individuals
            noise = np.random.uniform(-0.1, 0.1, population.shape)
            population += noise
            population = np.clip(population, self.lower_bound, self.upper_bound)

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
            self.preserve_diversity(population)

            best_idx = np.argmin(fitness)
            if budget_used < self.budget and np.random.rand() < 0.55:
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