import numpy as np

class EnhancedMemoryDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40  # Increased population size for more diversity
        self.bounds = [-5.0, 5.0]
        self.c1 = 2.0  # Enhanced cognitive parameter
        self.c2 = 2.0  # Enhanced social parameter
        self.w = 0.5  # Reduced inertia weight for quicker convergence
        self.F = 0.5  # Modified DE Mutation factor for better exploration
        self.CR = 0.6  # Modified DE Crossover probability
        self.memory = []  # Memory to store historical bests

    def dynamic_parameters(self, evaluations):
        # Adjust DE and PSO parameters dynamically based on function evaluations
        self.w = 0.9 - evaluations / self.budget * 0.4
        self.F = 0.4 + evaluations / self.budget * 0.1
        self.CR = 0.7 - evaluations / self.budget * 0.2

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
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
                    self.memory.append(global_best_position)

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
                if trial_score < personal_best_scores[i] or np.exp((personal_best_scores[i] - trial_score) / self.w) > np.random.rand():
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                memory_influence = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dim)
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + self.c1 * r1 * (personal_best_positions[i] - population[i]) + 
                                 self.c2 * r2 * (global_best_position - population[i]) + 0.1 * r3 * (memory_influence - population[i]))
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
                    self.memory.append(global_best_position)

        return global_best_position, global_best_score