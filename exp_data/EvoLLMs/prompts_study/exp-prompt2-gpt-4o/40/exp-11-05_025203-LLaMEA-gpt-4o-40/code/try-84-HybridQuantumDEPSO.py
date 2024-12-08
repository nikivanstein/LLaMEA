import numpy as np

class HybridQuantumDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40  # Increased population size for diversity
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.2  # Adjusted cognitive parameter
        self.c2 = 1.8  # Adjusted social parameter
        self.w = 0.7  # Modified inertia weight for improved exploration
        self.F = 0.7  # Balanced DE Mutation factor
        self.CR = 0.9  # Increased DE Crossover probability
        self.q = 0.5  # Quantum cloud radius initial
        self.alpha = 0.95  # Updated cooling rate

    def dynamic_parameters(self, evaluations):
        # Adjust DE and Quantum parameters dynamically
        self.F = 0.4 + (0.3 - evaluations / self.budget) * np.random.rand()
        self.CR = 0.7 + (0.3 * evaluations / self.budget) * np.random.rand()
        self.q *= self.alpha

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
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
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial

                if evaluations >= self.budget:
                    break

            # Quantum-behavior PSO update
            for i in range(self.pop_size):
                quantum_vector = np.random.normal(scale=self.q, size=self.dim)
                trial = personal_best_positions[i] + quantum_vector
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