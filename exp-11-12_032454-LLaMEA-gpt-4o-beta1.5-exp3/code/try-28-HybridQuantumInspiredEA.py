import numpy as np

class HybridQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(10, int(budget / (20 * dim)))  # heuristic for population size
        self.prob_mutation = 0.1  # mutation probability
        self.alpha = 0.5  # influence of parent solutions in differential crossover

    def __call__(self, func):
        # Initialize quantum bits as probabilistic positions in search space
        q_population = np.random.uniform(-np.pi, np.pi, (self.population_size, self.dim))
        population = self.quantum_to_real(q_population)
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                # Differential evolution-inspired crossover
                idxs = np.random.choice(self.population_size, 3, replace=False)
                r1, r2, r3 = q_population[idxs]
                q_trial = r1 + self.alpha * (r2 - r3)
                q_trial = np.clip(q_trial, -np.pi, np.pi)
                
                # Mutation
                if np.random.rand() < self.prob_mutation:
                    q_trial += np.random.normal(0, 0.1, self.dim)
                
                # Convert quantum representation to real values
                trial = self.quantum_to_real(q_trial)
                
                # Evaluate trial
                trial_fitness = func(trial)
                num_evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    q_population[i] = q_trial
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if num_evaluations >= self.budget:
                    break

        return best_solution, best_fitness

    def quantum_to_real(self, q_population):
        # Map the quantum population to real values in search space
        return (self.ub - self.lb) * (0.5 * (np.sin(q_population) + 1)) + self.lb