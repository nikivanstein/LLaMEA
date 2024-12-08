import numpy as np

class ImprovedHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def levy_flight(self, L=1.5):
        u = np.random.normal(scale=0.6966, size=self.dim)
        v = np.random.normal(scale=1.0, size=self.dim)
        step = u / np.power(np.abs(v), 1/L)
        return 0.01 * step

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size
        mutation_factor = np.ones(self.pop_size) * 0.5  # Initialize mutation factors
        
        while evals < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                self.F = mutation_factor[i]  # Use individual mutation factor
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.array([mutant[j] if np.random.rand() < self.CR else pop[i][j] for j in range(self.dim)])
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    mutation_factor[i] = min(mutation_factor[i] + 0.1, 1.0)  # Increase mutation factor
                else:
                    mutation_factor[i] = max(mutation_factor[i] - 0.1, 0.1)  # Decrease mutation factor
                
                if evals >= self.budget:
                    break
                
                # Crowding distance-based local search
                if evals < self.budget:
                    perturbation_size = 0.1 * (1.0 - evals / self.budget)
                    grad_perturb = np.random.randn(self.dim) * perturbation_size
                    local_trial = pop[i] + grad_perturb
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    
                    if local_fitness < fitness[i]:
                        pop[i] = local_trial
                        fitness[i] = local_fitness

                # Lévy flight for random exploration
                if evals < self.budget:
                    levy_step = self.levy_flight()
                    levy_trial = pop[i] + levy_step
                    levy_trial = np.clip(levy_trial, self.lower_bound, self.upper_bound)
                    levy_fitness = func(levy_trial)
                    evals += 1

                    if levy_fitness < fitness[i]:
                        pop[i] = levy_trial
                        fitness[i] = levy_fitness

            # Update crossover probability adaptively
            self.CR = 0.7 + 0.2 * np.random.rand()

            if evals >= self.budget:
                break
        
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]