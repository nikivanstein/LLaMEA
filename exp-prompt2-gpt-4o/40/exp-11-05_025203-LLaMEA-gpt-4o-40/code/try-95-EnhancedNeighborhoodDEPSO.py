import numpy as np

class EnhancedNeighborhoodDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40  # Adjusted population size for better exploration
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.7  # Slightly adjusted cognitive parameter
        self.c2 = 1.3  # Modified social parameter
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.F = 0.7  # Enhanced DE Mutation factor
        self.CR = 0.6  # Adjusted DE Crossover probability
        self.T = 1.0
        self.T_min = 1e-4  # Finer cooling adjustment
        self.alpha = 0.95  # Modified cooling rate for annealing

    def dynamic_parameters(self, evaluations):
        # Adjust parameters dynamically based on function evaluations
        self.F = 0.4 + (0.6 - evaluations / self.budget) * np.random.rand()
        self.CR = 0.4 + (0.5 - evaluations / self.budget) * np.random.rand()
        self.T = max(self.T_min, self.T * self.alpha)
        self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

    def neighborhood_search(self, individual, eval_func):
        # Local neighborhood search to enhance exploitation
        neighbor = np.clip(individual + np.random.normal(0, 0.1, self.dim), self.bounds[0], self.bounds[1])
        neighbor_score = eval_func(neighbor)
        return neighbor, neighbor_score

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
                velocities[i] = self.w * velocities[i] + self.c1 * r1 * (personal_best_positions[i] - population[i]) + self.c2 * r2 * (global_best_position - population[i])
                trial = population[i] + velocities[i]
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                trial_score = func(trial)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                # Apply neighborhood search to further refine solutions
                if evaluations < self.budget - 1:
                    neighbor, neighbor_score = self.neighborhood_search(population[i], func)
                    evaluations += 1
                    if neighbor_score < personal_best_scores[i]:
                        personal_best_scores[i] = neighbor_score
                        personal_best_positions[i] = neighbor
                        population[i] = neighbor

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score