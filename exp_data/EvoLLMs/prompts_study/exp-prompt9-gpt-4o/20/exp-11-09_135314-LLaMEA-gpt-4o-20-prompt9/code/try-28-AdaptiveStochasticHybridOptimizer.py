import numpy as np
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture

class AdaptiveStochasticHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(40, budget // 8)  # Adjusted population size
        self.strategy_switch = 0.25  # Switch to Differential Evolution after 25% of budget

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(func, 1, population)
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # Random Search with Gaussian Mixture model
                gmm = GaussianMixture(n_components=2, covariance_type='full')
                gmm.fit(population)
                samples = gmm.sample(self.population_size)[0]
                samples = np.clip(samples, self.lower_bound, self.upper_bound)
                for i in range(self.population_size):
                    candidate_fitness = func(samples[i])
                    evals += 1
                    if candidate_fitness < fitness[i]:
                        population[i] = samples[i]
                        fitness[i] = candidate_fitness
                        if candidate_fitness < best_fitness:
                            best_fitness = candidate_fitness
                            best_solution = samples[i].copy()
                    if evals >= self.budget:
                        break
                # Local refinement with Nelder-Mead
                if evals + self.dim + 1 <= self.budget:
                    result = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.dim + 1})
                    evals += result.nfev
                    if result.fun < best_fitness:
                        best_fitness = result.fun
                        best_solution = result.x
            else:
                # Differential Evolution with stochastic weight adjustment
                scale_factor = 0.5 + 0.4 * np.random.rand()
                for i in range(self.population_size):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    crossover_prob = np.random.rand(self.dim)
                    trial = np.where(crossover_prob < 0.85, mutant, population[i])
                    trial_fitness = func(trial)
                    evals += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best_solution = trial.copy()
                    if evals >= self.budget:
                        break

        return best_solution