import numpy as np

class HybridAdaptiveDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40  # Increased population size for diversity
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.7  # Enhanced cognitive parameter
        self.c2 = 1.7  # Enhanced social parameter
        self.w = 0.5  # Reduced inertia weight for faster convergence
        self.F = 0.9  # Increased DE Mutation factor for exploration
        self.CR = 0.6  # Modified DE Crossover probability
        self.oscillate_freq = 5  # New parameter for oscillation frequency
        self.T = 1.0
        self.T_min = 1e-3
        self.alpha = 0.85  # Modified cooling rate for temperature

    def dynamic_parameters(self, evaluations):
        # Adjust DE parameters dynamically based on function evaluations
        phase = (evaluations // self.oscillate_freq) % 2
        self.F = 0.6 + 0.3 * phase
        self.CR = 0.4 + 0.2 * phase
        self.T = max(self.T_min, self.T * self.alpha)

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))  # Zero initialization for velocities
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

                trial_score = func(trial)
                evaluations += 1
                if trial_score < personal_best_scores[i] or np.exp((personal_best_scores[i] - trial_score) / self.T) > np.random.rand():
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + self.c1 * r1 * (personal_best_positions[i] - population[i]) + self.c2 * r2 * (global_best_position - population[i]))
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