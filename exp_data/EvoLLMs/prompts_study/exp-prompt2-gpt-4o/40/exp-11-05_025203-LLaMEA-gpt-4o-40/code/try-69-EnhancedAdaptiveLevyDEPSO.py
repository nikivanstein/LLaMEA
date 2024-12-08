import numpy as np

class EnhancedAdaptiveLevyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30  # Population size
        self.bounds = [-5.0, 5.0]
        self.c1 = 2.0  # Modified cognitive parameter
        self.c2 = 2.0  # Modified social parameter
        self.w_max = 0.9  # Inertia weight max
        self.w_min = 0.4  # Inertia weight min
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.T = 1.0
        self.T_min = 1e-3
        self.alpha = 0.95  # Cooling rate

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, len(L))
        v = np.random.normal(0, 1, len(L))
        step = u / np.abs(v) ** (1 / beta)
        return 0.1 * step * (L - np.mean(L))

    def dynamic_parameters(self, evaluations):
        # Adjust DE parameters dynamically based on function evaluations
        self.F = 0.5 + 0.3 * np.random.rand()
        self.CR = 0.7 + 0.2 * np.random.rand()
        self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
        self.T = max(self.T_min, self.T * self.alpha)

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

            # Perform Lévy flight-based exploration
            levy_steps = self.levy_flight(global_best_position)
            for i in range(self.pop_size):
                new_position = personal_best_positions[i] + levy_steps
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
                new_score = func(new_position)
                evaluations += 1
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = new_position

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score