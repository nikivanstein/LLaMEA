import numpy as np

class AdaptiveLevyGaussianDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.bounds = [-5.0, 5.0]
        self.c1 = 2.0  # Increased cognitive parameter
        self.c2 = 1.5  # Retained social parameter
        self.w = 0.5  # Reduced inertia weight for better convergence
        self.F = 0.9  # Dynamic DE Mutation factor
        self.CR = 0.6  # Adjusted DE Crossover probability
        self.sigma = 0.1  # Standard deviation for Gaussian mutation

    def levy_flight(self, lam):
        u = np.random.normal(0, 1) * (0.6966 / np.abs(np.random.normal(0, 1)) ** (1 / lam))
        return u

    def dynamic_parameters(self, evaluations):
        self.F = 0.5 + (0.3 * (1 - evaluations / self.budget)) * np.random.rand()
        self.CR = 0.5 + (0.4 * (1 - evaluations / self.budget)) * np.random.rand()

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        
        while evaluations < self.budget:
            self.dynamic_parameters(evaluations)
            for i in range(self.pop_size):
                score = func(population[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]
                if evaluations >= self.budget:
                    break

            for i in range(self.pop_size):
                indices = [index for index in range(self.pop_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                if np.random.rand() < 0.5:  # Apply Lévy flights
                    step = self.levy_flight(1.5) * (population[i] - global_best_position)
                    trial += step

                trial = np.clip(trial, self.bounds[0], self.bounds[1])
                
                trial_score = func(trial)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i]))
                trial = population[i] + velocities[i] + self.sigma * np.random.normal(0, 1, self.dim)
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