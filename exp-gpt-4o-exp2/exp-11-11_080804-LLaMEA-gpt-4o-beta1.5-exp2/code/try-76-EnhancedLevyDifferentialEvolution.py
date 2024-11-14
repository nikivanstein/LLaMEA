import numpy as np

class EnhancedLevyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5
        self.cr = 0.9
        self.alpha = 1.5

    def levy_flight(self, size):
        u = np.random.normal(0, 1, size) * (np.sqrt(np.abs(np.random.normal(0, 1, size))) ** (-1 / self.alpha))
        return u

    def adaptive_mutation_scaling(self, evaluations, temperature):
        return self.f * (1 - (evaluations / self.budget)) * temperature * (1 + np.random.normal(0, 0.01))

    def dynamic_crossover_probability(self, evaluations):
        return self.cr * (0.7 + 0.3 * (self.budget - evaluations) / self.budget) * (1 + np.random.normal(0, 0.01))

    def adaptive_noise_level(self, evaluations):
        return 0.1 * (1 - evaluations / self.budget)

    def temperature_factor(self, evaluations):
        return 0.5 + 0.5 * (1 - evaluations / self.budget)

    def adaptive_shrinkage_factor(self, evaluations):
        return (1 - evaluations / self.budget) ** 0.5

    def stochastic_perturbation(self, evaluations):
        return np.random.uniform(-0.1, 0.1, self.dim) * (1 - evaluations / self.budget)

    def adaptive_learning_rate(self, evaluations):
        return 0.1 + 0.9 * (1 - evaluations / self.budget)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            elite_size = max(1, int(3 * (1 - evaluations / self.budget)))
            elite = population[np.argsort(fitness)[:elite_size]]
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                temperature = self.temperature_factor(evaluations)
                adaptive_f = self.adaptive_mutation_scaling(evaluations, temperature)
                noise_level = self.adaptive_noise_level(evaluations)
                noise = np.random.normal(0, noise_level, self.dim)
                shrink_factor = self.adaptive_shrinkage_factor(evaluations)

                local_best = elite[np.random.randint(0, elite_size)]
                mutant = np.clip(x0 + adaptive_f * (x1 - x2 + noise) * shrink_factor + 0.1 * (local_best - x0), self.lower_bound, self.upper_bound)

                crossover_prob = self.dynamic_crossover_probability(evaluations)
                crossover = np.random.rand(self.dim) < crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                adaptive_lr = self.adaptive_learning_rate(evaluations)
                stochastic_perturb = self.stochastic_perturbation(evaluations)
                trial += adaptive_lr * (stochastic_perturb + self.levy_flight(self.dim)) * np.exp(-evaluations / self.budget)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best = trial

                if evaluations >= self.budget:
                    break

        return best