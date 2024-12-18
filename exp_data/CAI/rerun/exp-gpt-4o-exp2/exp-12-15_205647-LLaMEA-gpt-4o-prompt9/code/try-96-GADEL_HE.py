import numpy as np
from sklearn.ensemble import RandomForestRegressor

class GADEL_HE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(60, budget // 8)
        self.F = 0.5
        self.CR = 0.9
        self.learning_rate = 0.1
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        best_idx = np.argmin(fitness)
        best = population[best_idx]

        while eval_count < self.budget:
            # Train surrogate model intermittently
            if eval_count % (self.population_size * 4) == 0:
                self.model.fit(population, fitness)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                F_dynamic = (self.F + np.random.normal(0, 0.05)) * (1 + 0.05 * np.exp(-eval_count / (self.budget / 3)))
                mutant = np.clip(x0 + F_dynamic * (x1 - x2), self.lower_bound, self.upper_bound)

                CR_dynamic = self.CR + 0.1 * np.cos(eval_count / self.budget * np.pi)
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                
                # Use surrogate prediction with small probability
                if np.random.rand() < 0.1 and hasattr(self.model, 'predict'):
                    f_trial = self.model.predict(trial.reshape(1, -1))[0]
                else:
                    f_trial = func(trial)
                    eval_count += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial

            if eval_count % (self.population_size * 2) == 0:
                self.F = 0.4 + 0.25 * (eval_count / self.budget)
                self.CR = 0.9 - 0.35 * (eval_count / self.budget)

            neighborhood_scale = np.clip(0.8 - 0.6 * (eval_count / self.budget), 0.2, 0.8)
            neighborhood = np.clip(best + np.random.normal(0, self.learning_rate * neighborhood_scale, self.dim), self.lower_bound, self.upper_bound)
            f_neighborhood = func(neighborhood)
            eval_count += 1
            if f_neighborhood < fitness[best_idx]:
                best = neighborhood
                fitness[best_idx] = f_neighborhood

        return best, fitness[best_idx]