import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30  # Increased population size
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.8  # Adjusted cognitive parameter
        self.c2 = 1.8  # Adjusted social parameter
        self.w = 0.4  # Updated inertia weight to increase exploration
        self.F = 0.5  # Modified DE Mutation factor
        self.CR = 0.8  # Enhanced DE Crossover probability
        self.T = 1.0 
        self.T_min = 1e-3  # New minimum temperature
        self.alpha = 0.9  # Enhanced cooling rate

    def adaptive_update(self, evaluations):
        self.F = 0.5 * (1 - evaluations / self.budget) + 0.2 * np.random.rand()
        self.CR = 0.5 + 0.4 * np.random.rand()
        self.T = max(self.T_min, self.T * self.alpha)  # Enhanced exponential cooling

    def competition_based_diversity(self, population, scores):
        sorted_indices = np.argsort(scores)
        return population[sorted_indices[:self.pop_size // 2]]

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))  # Random initial velocities
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        
        while evaluations < self.budget:
            self.adaptive_update(evaluations)
            scores = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                score = func(population[i])
                evaluations += 1
                scores[i] = score
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]
                
                if evaluations >= self.budget:
                    break

            # Enhance diversity by keeping top half of population
            population = self.competition_based_diversity(population, scores)
            population = np.concatenate((population, np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size // 2, self.dim))))

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