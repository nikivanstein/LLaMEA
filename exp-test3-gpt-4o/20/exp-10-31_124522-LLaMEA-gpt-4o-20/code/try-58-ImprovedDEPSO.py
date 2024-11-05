import numpy as np

class ImprovedDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.w = 0.7  # Inertia weight for PSO
        self.c1 = 1.5 # Cognitive component for PSO
        self.c2 = 1.5 # Social component for PSO
        self.F = 0.8  # Differential weight for DE
        self.CR = 0.9 # Crossover probability for DE
        
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility

        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        
        fitness = np.apply_along_axis(func, 1, pop)
        personal_best_positions = np.copy(pop)
        personal_best_fitness = np.copy(fitness)
        global_best_position = pop[np.argmin(fitness)]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            self.w = max(0.2, self.w * (1 - evaluations / self.budget))  # Dynamic inertia weight
            self.F = 0.5 + 0.5 * (1 - evaluations / self.budget) * np.random.rand()
            self.CR = 0.5 + 0.5 * np.random.rand()

            dynamic_size = max(10, int(self.population_size * (1 - evaluations / self.budget)))
            current_population_size = min(self.population_size, dynamic_size)
            
            mutation_factor = self.F + 0.1 * np.sin(np.pi * evaluations / self.budget)

            ranks = np.argsort(fitness)
            for i in range(current_population_size):
                idxs = [idx for idx in range(current_population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < func(global_best_position):
                            global_best_position = trial

                if evaluations >= self.budget:
                    break
            
            for i in range(current_population_size):
                velocities[i] = (self.w * velocities[i] 
                                 + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - pop[i])
                                 + self.c2 * np.random.rand(self.dim) * (global_best_position - pop[i]))
                pop[i] = np.clip(pop[i] + velocities[i], self.lower_bound, self.upper_bound)
                
                current_fitness = func(pop[i])
                evaluations += 1
                if current_fitness < fitness[i]:
                    fitness[i] = current_fitness
                    personal_best_positions[i] = pop[i]
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < func(global_best_position):
                        global_best_position = pop[i]
                
                if evaluations >= self.budget:
                    break
        
            elite_idx = np.argmin(fitness)  # Elitism: Retain the best solution
            pop[elite_idx] = global_best_position
            fitness[elite_idx] = func(global_best_position)

        return global_best_position, func(global_best_position)