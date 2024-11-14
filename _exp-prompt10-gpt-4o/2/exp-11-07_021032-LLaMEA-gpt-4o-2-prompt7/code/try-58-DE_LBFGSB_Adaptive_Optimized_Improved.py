import numpy as np
from scipy.optimize import minimize
import concurrent.futures

class DE_LBFGSB_Adaptive_Optimized_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.f = 0.8
        self.cr = 0.9

    def __call__(self, func):
        population = self.initialize_population()
        scores = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.evaluate_individual, func, population, scores, i) 
                           for i in range(self.pop_size)]
                for future in concurrent.futures.as_completed(futures):
                    trial, trial_score, i = future.result()
                    if evaluations >= self.budget:
                        break
                    evaluations += 1
                    if trial_score < scores[i]:
                        population[i] = trial
                        scores[i] = trial_score

            self.f = 0.6 + 0.2 * np.random.rand()
            self.cr = 0.7 + 0.3 * np.random.rand()

            if evaluations < self.budget:
                best_idx = np.argmin(scores)
                weights = np.exp(-scores / np.sum(scores))
                weighted_average = np.dot(population.T, weights) / np.sum(weights)
                result = minimize(func, weighted_average, method='L-BFGS-B',
                                  bounds=[self.bounds] * self.dim,
                                  options={'maxfun': self.budget - evaluations})
                lbfgsb_evals = result.nfev
                evaluations += lbfgsb_evals
                if result.fun < scores[best_idx]:
                    population[best_idx] = result.x
                    scores[best_idx] = result.fun

        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))

    def evaluate_individual(self, func, population, scores, idx):
        target = population[idx]
        mutant = self.mutation(population, scores, idx)
        trial = self.crossover(mutant, target)
        trial_score = func(trial)
        return trial, trial_score, idx

    def mutation(self, population, scores, idx):
        candidates = np.argsort(scores)
        top_candidates = candidates[:self.pop_size // 2]
        a, b, c = np.random.choice(top_candidates[top_candidates != idx], 3, replace=False)
        mutant = population[a] + self.f * (population[b] - population[c])
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def crossover(self, mutant, target):
        cross_points = np.random.rand(self.dim) < self.cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial