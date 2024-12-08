import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.bounds = [-5.0, 5.0]
        self.c1_initial, self.c2_initial = 1.5, 1.5  # Initial PSO parameters
        self.F_initial, self.CR_initial = 0.8, 0.8  # Initial DE parameters
        self.T = 1.0
        self.learning_factor = 0.99  # Adaptive learning rate

    def levy_flight(self, step_size):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1/3)
        return step_size * step

    def adaptive_update(self, evaluations):
        # Adaptive strategy for parameters and cooling schedule
        self.F = self.F_initial * np.random.rand()
        self.CR = self.CR_initial * np.random.rand()
        self.T *= 0.95  # Enhanced cooling rate
        self.c1 = self.c1_initial * (self.learning_factor ** evaluations)
        self.c2 = self.c2_initial * (self.learning_factor ** evaluations)

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')
        evaluations = 0
        
        while evaluations < self.budget:
            self.adaptive_update(evaluations)
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
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.F * (b - c) + self.levy_flight(0.01)  # Added Lévy flight
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
                velocities[i] = (self.c1 * r1 * (personal_best_positions[i] - population[i]) +
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