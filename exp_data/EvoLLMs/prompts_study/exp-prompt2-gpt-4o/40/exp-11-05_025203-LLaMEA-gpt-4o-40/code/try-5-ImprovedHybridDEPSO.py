import numpy as np

class ImprovedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.bounds = [-5.0, 5.0]  # Search space bounds
        self.c1 = 1.5  # PSO cognitive parameter (tuned)
        self.c2 = 1.5  # PSO social parameter (tuned)
        self.w = 0.4  # Inertia weight for PSO (tuned)
        self.F = 0.8  # DE Mutation factor (adaptive)
        self.CR = 0.8  # DE Crossover probability (adaptive)

    def adapt_parameters(self, evaluations):
        # Adaptive strategy for parameters
        self.F = 0.5 + 0.3 * np.random.rand()
        self.CR = 0.5 + 0.4 * np.random.rand()

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        
        while evaluations < self.budget:
            self.adapt_parameters(evaluations)  # Adaptive parameter adjustment
            # Evaluate population
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

            # Apply DE/PSO hybridization
            for i in range(self.pop_size):
                # Differential Evolution
                indices = [index for index in range(self.pop_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                # Particle Swarm Optimization
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) + 
                                 self.c2 * r2 * (global_best_position - population[i]))
                trial += velocities[i]
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                trial_score = func(trial)
                evaluations += 1

                # Greedy selection
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