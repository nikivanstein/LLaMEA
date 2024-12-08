import numpy as np

class HybridPSOWithAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.F_base = 0.5  # base scaling factor for mutation
        self.CR = 0.9  # crossover probability
        
    def __call__(self, func):
        # Initialize swarm
        positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = positions.copy()
        personal_best_fitness = np.array([func(ind) for ind in positions])
        num_evaluations = self.population_size
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - positions[i]) +
                                 self.c2 * r2 * (global_best - positions[i]))
                
                # Update positions
                positions[i] = np.clip(positions[i] + velocities[i], self.lb, self.ub)
                
                # Evaluate current position
                fitness = func(positions[i])
                num_evaluations += 1
                
                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = positions[i]
                    personal_best_fitness[i] = fitness
                
                # Update global best
                if fitness < global_best_fitness:
                    global_best = positions[i]
                    global_best_fitness = fitness
            
            # Adaptive mutation phase
            if num_evaluations < self.budget:
                for i in range(self.population_size):
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b = np.random.choice(indices, 2, replace=False)
                    F = self.F_base + np.random.rand() * (1.0 - self.F_base)
                    mutated_vector = personal_best[a] + F * (personal_best[b] - personal_best[i])
                    mutated_vector = np.clip(mutated_vector, self.lb, self.ub)
                    
                    # Crossover
                    crossover_mask = np.random.rand(self.dim) < self.CR
                    trial_vector = np.where(crossover_mask, mutated_vector, positions[i])
                    
                    # Evaluate trial vector
                    trial_fitness = func(trial_vector)
                    num_evaluations += 1
                    
                    # Replace if better
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial_vector
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < global_best_fitness:
                            global_best = trial_vector
                            global_best_fitness = trial_fitness
        
        return global_best, global_best_fitness