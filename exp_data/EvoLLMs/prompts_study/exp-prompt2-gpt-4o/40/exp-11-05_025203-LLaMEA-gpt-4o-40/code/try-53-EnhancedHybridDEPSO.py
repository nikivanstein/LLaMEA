import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30  # Slightly increased population size
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.5  # Refined cognitive parameter
        self.c2 = 1.5  # Refined social parameter
        self.w = 0.6  # Adjusted inertia weight for better exploration
        self.F = 0.8  # Enhanced DE Mutation factor
        self.CR = 0.85  # Refined DE Crossover probability
        self.T = 1.0 
        self.T_min = 1e-3  # Minimum temperature remains the same
        self.alpha = 0.90  # Adjusted cooling rate for slower annealing

    def adaptive_update(self, evaluations):
        self.F = 0.5 + 0.3 * np.random.rand()  # More dynamic mutation factor
        self.CR = 0.7 + 0.2 * np.random.rand()  # More dynamic crossover rate
        self.T = max(self.T_min, self.T * self.alpha)  # Exponential cooling remains

    def chaotic_local_search(self, position):
        # Introduce chaotic perturbation for local search
        return position + 0.01 * np.sin(10 * np.pi * position) * np.random.normal(0, 1, self.dim)
    
    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        topology = np.random.randint(0, self.pop_size, (self.pop_size, 3))  # Dynamic neighborhood

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
                indices = topology[i]
                a, b, c = population[indices]
                mutant = a + self.F * (b - c)
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
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) + 
                                 self.c2 * r2 * (global_best_position - population[i]))
                trial = population[i] + velocities[i]
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                # Apply chaotic local search occasionally
                if np.random.rand() < 0.1:
                    trial = self.chaotic_local_search(trial)

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