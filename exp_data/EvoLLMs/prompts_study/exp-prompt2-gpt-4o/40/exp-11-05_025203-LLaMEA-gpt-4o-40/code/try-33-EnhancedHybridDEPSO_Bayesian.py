import numpy as np
from scipy.stats import norm

class EnhancedHybridDEPSO_Bayesian:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.6
        self.F = 0.8
        self.CR = 0.9
        self.T = 1.0
        self.kernel_noise = 1e-6  # Small noise to ensure numerical stability

    def adaptive_update(self, evaluations):
        self.F = 0.5 + 0.4 * np.random.rand()
        self.CR = 0.5 + 0.4 * np.random.rand()
        self.T *= 0.85

    def gaussian_process(self, X, y, X_star):
        K = np.exp(-0.5 * np.subtract.outer(X, X)**2) + self.kernel_noise * np.eye(len(X))
        K_star = np.exp(-0.5 * np.subtract.outer(X_star, X)**2)
        K_star_star = np.exp(-0.5 * np.subtract.outer(X_star, X_star)**2)
        K_inv = np.linalg.inv(K)
        
        mu_star = K_star @ K_inv @ y
        cov_star = K_star_star - K_star @ K_inv @ K_star.T
        return mu_star, cov_star

    def expected_improvement(self, X, y, X_star, xi=0.01):
        mu_star, sigma_star = self.gaussian_process(X, y, X_star)
        sigma_star = np.sqrt(np.diag(sigma_star))
        y_min = np.min(y)
        z = (y_min - mu_star - xi) / sigma_star
        ei = (y_min - mu_star - xi) * norm.cdf(z) + sigma_star * norm.pdf(z)
        return ei

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0

        eval_points = []
        eval_scores = []

        while evaluations < self.budget:
            self.adaptive_update(evaluations)
            
            for i in range(self.pop_size):
                score = func(population[i])
                evaluations += 1
                eval_points.append(population[i])
                eval_scores.append(score)
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]
                
                if evaluations >= self.budget:
                    break

            # Bayesian Optimization Exploration
            if evaluations + self.pop_size <= self.budget:
                X = np.array(eval_points)
                y = np.array(eval_scores)
                X_star = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
                ei = self.expected_improvement(X, y, X_star)
                new_indices = np.argsort(-ei)  # Select based on max EI

                for i in range(self.pop_size):
                    suggested = X_star[new_indices[i]]
                    suggested_score = func(suggested)
                    evaluations += 1
                    if suggested_score < global_best_score:
                        global_best_score = suggested_score
                        global_best_position = suggested
                    eval_points.append(suggested)
                    eval_scores.append(suggested_score)
                    if evaluations >= self.budget:
                        break

            for i in range(self.pop_size):
                indices = [index for index in range(self.pop_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                trial_score = func(trial)
                evaluations += 1
                if (trial_score < personal_best_scores[i] or 
                    np.exp((personal_best_scores[i] - trial_score) / self.T) > np.random.rand()):
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) + 
                                 self.c2 * r2 * (global_best_position - population[i]))
                trial = population[i] + velocities[i]
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                trial_score = func(trial)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score