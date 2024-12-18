import numpy as np

class GADEL_EL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(60, budget // 8)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.elite_ratio = 0.1
        self.adaptive_rate = 0.1

    def __call__(self, func):
        np.random.seed(42)
        
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        num_elites = max(1, int(self.elite_ratio * self.population_size))
        elites = np.argpartition(fitness, num_elites)[:num_elites]
        best_idx = np.argmin(fitness)
        best = population[best_idx]

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                intensity_factor = 1 + 0.05 * np.cos(2 * np.pi * eval_count / self.budget)
                F_dynamic = (self.F + np.random.normal(0, 0.03)) * intensity_factor
                mutant = np.clip(x0 + F_dynamic * (x1 - x2), self.lower_bound, self.upper_bound)

                CR_dynamic = self.CR + 0.05 * np.sin(eval_count / self.budget * np.pi)
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial

            adapt_factor = 0.2 * (1 - eval_count / self.budget)
            self.F = max(0.4, self.F - self.adaptive_rate * adapt_factor)
            self.CR = min(0.95, self.CR + self.adaptive_rate * adapt_factor)
            
            # Elite preservation and exploration
            for e in elites:
                neighborhood = np.clip(population[e] + np.random.normal(0, adapt_factor, self.dim), self.lower_bound, self.upper_bound)
                f_neighborhood = func(neighborhood)
                eval_count += 1
                if f_neighborhood < fitness[e]:
                    population[e] = neighborhood
                    fitness[e] = f_neighborhood
                    if f_neighborhood < fitness[best_idx]:
                        best = neighborhood
                        fitness[best_idx] = f_neighborhood

        return best, fitness[best_idx]