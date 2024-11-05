import numpy as np

class EnhancedDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25  # Increased population size
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.7  # Updated PSO cognitive parameter
        self.c2 = 1.7  # Updated PSO social parameter
        self.w = 0.5  # Updated inertia weight for PSO
        self.F = 0.9  # More aggressive DE mutation factor
        self.CR = 0.9  # Higher DE crossover probability

    def adapt_parameters(self, evaluations, max_evals):
        # Adaptive strategy for parameters based on progress
        self.F = 0.5 + 0.4 * (1 - evaluations / max_evals) * np.random.rand()
        self.CR = 0.5 + 0.5 * (evaluations / max_evals) * np.random.rand()

    def local_random_search(self, candidate):
        # Introduce a small local random search step
        return candidate + 0.1 * np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        
        while evaluations < self.budget:
            self.adapt_parameters(evaluations, self.budget)
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

                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) + 
                                 self.c2 * r2 * (global_best_position - population[i]))
                trial += velocities[i]
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                trial = self.local_random_search(trial)  # Apply local search
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