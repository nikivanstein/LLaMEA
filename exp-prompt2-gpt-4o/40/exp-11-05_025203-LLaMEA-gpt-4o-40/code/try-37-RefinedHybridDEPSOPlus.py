import numpy as np

class RefinedHybridDEPSOPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.bounds = [-5.0, 5.0]  # Search space bounds
        self.c1 = 1.5  # PSO cognitive parameter
        self.c2 = 1.5  # PSO social parameter
        self.w = 0.5  # Inertia weight for PSO (adapted)
        self.F = 0.9  # DE Mutation factor (adapted)
        self.CR = 0.9  # DE Crossover probability (adapted)
        self.T = 1.0  # Initial temperature for Simulated Annealing

    def dynamic_update(self, evaluations):
        # Dynamic strategy for parameters and cooling schedule
        self.F = 0.5 + 0.4 * np.random.rand()
        self.CR = 0.7 + 0.3 * np.random.rand()
        self.T *= 0.95  # Slower cooling

    def orthogonal_crossover(self, parent1, parent2):
        # Generate offspring using orthogonal crossover
        offspring = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                offspring[i] = parent1[i]
            else:
                offspring[i] = parent2[i]
        return np.clip(offspring, self.bounds[0], self.bounds[1])

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0

        while evaluations < self.budget:
            self.dynamic_update(evaluations)
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
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = self.orthogonal_crossover(population[i], mutant)
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