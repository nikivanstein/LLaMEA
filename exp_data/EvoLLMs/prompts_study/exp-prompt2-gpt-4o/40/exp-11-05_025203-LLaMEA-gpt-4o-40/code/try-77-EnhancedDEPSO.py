import numpy as np

class EnhancedDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.7  # Slightly increased cognitive parameter
        self.c2 = 1.7  # Slightly increased social parameter
        self.w_min = 0.4  # Minimum inertia weight
        self.w_max = 0.9  # Maximum inertia weight
        self.F = 0.5  # Initial DE Mutation factor
        self.CR = 0.5  # Initial DE Crossover probability

    def adaptive_parameters(self, evaluations):
        # Self-adapt parameters based on current evaluations
        self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
        self.F = 0.5 + 0.5 * np.random.rand()
        self.CR = 0.5 + 0.5 * np.random.rand()

    def elitist_mutation(self, population, global_best_position):
        # Perform elitist mutation with the global best
        indices = np.arange(self.pop_size)
        np.random.shuffle(indices)
        for i in range(0, self.pop_size, 3):
            if i + 2 < self.pop_size:
                a, b, c = indices[i:i+3]
                mutant = population[a] + self.F * (population[b] - population[c]) + 0.1 * (global_best_position - population[a])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                if np.random.rand() < self.CR:
                    population[a] = mutant

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        
        while evaluations < self.budget:
            self.adaptive_parameters(evaluations)
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

            self.elitist_mutation(population, global_best_position)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + self.c1 * r1 * (personal_best_positions[i] - population[i]) +
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