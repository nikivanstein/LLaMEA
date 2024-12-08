import numpy as np
from sklearn.cluster import KMeans

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(50, max(10, budget // (8 * dim)))
        self.f = np.random.uniform(0.4, 0.9)  # Dynamically adjusted DE scaling factor
        self.cr = 0.8 + 0.1 * np.random.rand()  # Slightly reduced crossover probability
        self.w = np.random.uniform(0.3, 0.6)  # Adaptive inertia weight range
        self.c1 = np.random.uniform(1.5, 2.5)  # Adaptive cognitive coefficient
        self.c2 = np.random.uniform(1.0, 2.0)  # Adaptive social coefficient
        self.v_max = (self.upper_bound - self.lower_bound) / 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.pop_size

        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx].copy()
        
        while budget_used < self.budget:
            for i in range(self.pop_size):
                if budget_used >= self.budget:
                    break
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                if np.random.rand() < 0.7:
                    mutant = np.clip(a + np.random.uniform(0.5, 1.0) * (b - c), self.lower_bound, self.upper_bound)
                else:
                    mutant = np.clip(global_best + np.random.uniform(0.5, 1.0) * (b - a), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < personal_best_fitness[global_best_idx]:
                            global_best = trial
                            global_best_idx = i

            self.c1 = np.random.uniform(1.5, 2.5) * (1 - budget_used / self.budget)
            self.c2 = np.random.uniform(1.0, 2.0) * (budget_used / self.budget)

            historic_best = fitness.min()
            self.w = 0.4 + 0.5 * (historic_best - fitness.mean()) / (historic_best + 1e-9)
            self.w = np.clip(self.w, 0.1, 0.6)

            kmeans = KMeans(n_clusters=3, random_state=0).fit(population)  # Clustering to enhance exploration
            cluster_centers = kmeans.cluster_centers_

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = np.clip(
                self.w * velocities 
                + self.c1 * r1 * (personal_best - population) 
                + self.c2 * r2 * (global_best - cluster_centers[kmeans.labels_]),
                -self.v_max, self.v_max
            )
            population = np.clip(population + velocities, self.lower_bound, self.upper_bound)
            for i in range(self.pop_size):
                if budget_used >= self.budget:
                    break
                new_fitness = func(population[i])
                budget_used += 1
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < personal_best_fitness[global_best_idx]:
                        global_best = population[i]
                        global_best_idx = i

        return global_best