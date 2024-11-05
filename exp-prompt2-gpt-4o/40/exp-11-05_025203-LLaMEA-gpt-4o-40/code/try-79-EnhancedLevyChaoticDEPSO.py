import numpy as np

class EnhancedLevyChaoticDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Increased population size for diversity
        self.bounds = [-5.0, 5.0]
        self.c1 = 2.0  # Adjusted cognitive parameter
        self.c2 = 2.0  # Adjusted social parameter
        self.w = 0.5  # Reduced inertia weight for better exploitation
        self.F = 0.5  # Reduced DE Mutation factor for stability
        self.CR = 0.9  # Increased DE Crossover probability for higher genetic diversity
        self.T = 1.0
        self.T_min = 1e-4
        self.alpha = 0.95  # Adjusted cooling rate
        self.beta = 1.5  # Levy flight exponent

    def levy_flight(self):
        # Levy flight step
        u = np.random.normal(0, 1, self.dim) * self.F
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1/self.beta))
        return step

    def chaotic_map(self, evaluations):
        # Chaotic map for parameter adaptation
        return np.random.rand() * (1 - evaluations / self.budget)

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        
        while evaluations < self.budget:
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

                trial += self.levy_flight()
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < personal_best_scores[i] or np.exp((personal_best_scores[i] - trial_score) / self.T) > np.random.rand():
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                r1, r2 = np.random.rand(), self.chaotic_map(evaluations)
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