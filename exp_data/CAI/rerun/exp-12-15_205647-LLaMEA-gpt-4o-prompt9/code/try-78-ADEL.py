import numpy as np

class ADEL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(60, budget // 8)
        self.F = 0.5
        self.CR = 0.9
        self.learning_rate = 0.1

    def __call__(self, func):
        np.random.seed(42)
        
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        best_idx = np.argmin(fitness)
        best = population[best_idx]

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                # Adaptive differential weight
                self.F = 0.5 + 0.3 * np.tanh((fitness[i] - fitness[best_idx]) / abs(fitness[best_idx]) + 1e-10)
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.CR
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

            if eval_count % (self.population_size * 3) == 0:
                # Redistribute population
                new_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size // 4, self.dim))
                new_population_fitness = np.array([func(ind) for ind in new_population])
                eval_count += self.population_size // 4
                # Replace worst individuals
                worst_indices = np.argsort(fitness)[-self.population_size // 4:]
                population[worst_indices] = new_population
                fitness[worst_indices] = new_population_fitness

            neighborhood_scale = np.clip(0.8 - 0.6 * (eval_count / self.budget), 0.2, 0.8)
            neighborhood = np.clip(best + np.random.normal(0, self.learning_rate * neighborhood_scale, self.dim), self.lower_bound, self.upper_bound)
            f_neighborhood = func(neighborhood)
            eval_count += 1
            if f_neighborhood < fitness[best_idx]:
                best = neighborhood
                fitness[best_idx] = f_neighborhood

        return best, fitness[best_idx]