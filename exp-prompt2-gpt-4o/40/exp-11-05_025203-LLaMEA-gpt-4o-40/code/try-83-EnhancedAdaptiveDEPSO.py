import numpy as np

class EnhancedAdaptiveDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40  # Increased population size for better diversity
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.6  # Tweaked cognitive parameter
        self.c2 = 1.4  # Tweaked social parameter
        self.w = 0.7  # Adjusted inertia weight for better convergence
        self.F = 0.9  # Improved DE Mutation factor
        self.CR = 0.6  # Adjusted DE Crossover probability
        self.T = 1.0
        self.T_min = 1e-3
        self.alpha = 0.95  # Enhanced cooling rate
        self.velocity_clamp = 0.1  # Introduced max velocity to prevent explosion

    def dynamic_parameters(self, evaluations):
        # Adjust DE parameters dynamically based on function evaluations
        self.F = 0.4 + (0.2 * evaluations / self.budget) * np.random.rand()
        self.CR = 0.3 + (0.5 * evaluations / self.budget) * np.random.rand()
        self.T = max(self.T_min, self.T * self.alpha)

    def fitness_proportional_selection(self, scores):
        # Select indices based on fitness proportional selection
        total_fitness = np.sum(1.0 / (1.0 + np.array(scores)))
        probabilities = (1.0 / (1.0 + np.array(scores))) / total_fitness
        return np.random.choice(range(self.pop_size), p=probabilities)

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
                # Fitness proportional selection
                a = self.fitness_proportional_selection(personal_best_scores)
                b = self.fitness_proportional_selection(personal_best_scores)
                c = self.fitness_proportional_selection(personal_best_scores)
                
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
                velocities[i] = np.clip(self.w * velocities[i] + self.c1 * r1 * (personal_best_positions[i] - population[i]) + self.c2 * r2 * (global_best_position - population[i]), -self.velocity_clamp, self.velocity_clamp)
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