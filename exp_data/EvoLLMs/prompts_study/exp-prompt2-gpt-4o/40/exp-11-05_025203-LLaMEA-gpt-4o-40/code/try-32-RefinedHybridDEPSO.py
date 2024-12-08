import numpy as np

class RefinedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.bounds = [-5.0, 5.0]  # Search space bounds
        self.c1 = 1.5  # PSO cognitive parameter (tuned)
        self.c2 = 1.5  # PSO social parameter (tuned)
        self.w = 0.4  # Inertia weight for PSO (tuned)
        self.F = 0.8  # DE Mutation factor
        self.CR = 0.8  # DE Crossover probability
        self.T = 1.0  # Initial temperature for Simulated Annealing
        self.elite_fraction = 0.2  # Fraction of elite individuals retained

    def levy_flight(self, L):
        # Levy flight for random search with heavy-tailed probability distribution
        sigma = (np.math.gamma(1 + L) * np.sin(np.pi * L / 2) /
                 (np.math.gamma((1 + L) / 2) * L * 2 ** ((L - 1) / 2))) ** (1 / L)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def adaptive_update(self, evaluations):
        self.F = 0.5 + 0.3 * np.random.rand()
        self.CR = 0.5 + 0.4 * np.random.rand()
        self.T *= 0.95  # Exponential cooling

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
            scores = np.apply_along_axis(func, 1, population)
            evaluations += self.pop_size
            
            for i, score in enumerate(scores):
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]
            
            elites = population[np.argsort(personal_best_scores)[:int(self.elite_fraction * self.pop_size)]]
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                indices = [index for index in range(self.pop_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutant = a + self.F * (b - c) + self.levy_flight(1.5)  # Levy flight enhanced mutation
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

        return global_best_position, global_best_score