import numpy as np

class SpiralEnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5
        self.cr = 0.9
        self.alpha = 1.5

    def levy_flight(self, size, evaluations):
        scale_factor = 1 - evaluations / self.budget
        u = np.random.normal(0, 1, size) * (np.abs(np.random.normal(0, 1, size)) ** (-1 / self.alpha)) * scale_factor
        return u

    def adaptive_mutation_scaling(self, evaluations, temperature):
        return self.f * (1 - (evaluations / self.budget)) * temperature * (1 + np.random.normal(0, 0.01))

    def dynamic_crossover_probability(self, evaluations):
        return self.cr * (0.7 + 0.3 * (self.budget - evaluations) / self.budget) * (1 + np.random.normal(0, 0.01))

    def temperature_factor(self, evaluations):
        return 0.5 + 0.5 * (1 - evaluations / self.budget)
    
    def spiral_update(self, base, target, evaluations):
        angle = 0.2 * np.pi * (1 - evaluations / self.budget)
        spiral_factor = 1.0 + angle
        return base + spiral_factor * (target - base)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        
        while evaluations < self.budget:
            elite_size = max(1, int(2 + 3 * (1 - evaluations / self.budget)))
            elite = population[np.argsort(fitness)[:elite_size]]
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                temperature = self.temperature_factor(evaluations)
                adaptive_f = self.adaptive_mutation_scaling(evaluations, temperature)

                local_best = elite[np.random.randint(0, elite_size)]
                mutant = np.clip(x0 + adaptive_f * (x1 - x2) + 0.15 * (local_best - x0), self.lower_bound, self.upper_bound)
                trial = self.spiral_update(mutant, population[i], evaluations)
                
                crossover_prob = self.dynamic_crossover_probability(evaluations)
                crossover = np.random.rand(self.dim) < crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, trial, population[i])
                
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