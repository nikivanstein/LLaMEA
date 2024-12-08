import numpy as np

class EnhancedCrowdingMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.elite_archive_size = 0.1 * self.pop_size

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size
        mutation_factor = np.ones(self.pop_size) * 0.5
        elite_archive = np.empty((0, self.dim))

        while evals < self.budget:
            crowding = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                distances = np.linalg.norm(pop[idxs] - pop[i], axis=1)
                crowding[i] = np.sum(distances)

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                crowding_factor = 1 + (np.max(crowding) - crowding[i]) / np.max(crowding)
                self.F = mutation_factor[i] * crowding_factor
                self.CR = 0.9 * crowding_factor 
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                trial = np.array([mutant[j] if np.random.rand() < self.CR else
                                  (pop[i][j] + mutant[j]) / 2 for j in range(self.dim)])
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    mutation_factor[i] = min(mutation_factor[i] + 0.1, 1.0)
                    elite_archive = np.vstack((elite_archive, trial))
                    if len(elite_archive) > self.elite_archive_size:
                        elite_archive = elite_archive[np.argsort([func(e) for e in elite_archive])[:self.elite_archive_size]]
                else:
                    mutation_factor[i] = max(mutation_factor[i] - 0.1, 0.1)

                if evals >= self.budget:
                    break

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

            if evals >= self.budget:
                break
            
            if len(elite_archive) > 0:
                random_index = np.random.choice(len(elite_archive))
                random_elite = elite_archive[random_index]
                pop[np.random.randint(self.pop_size)] = random_elite

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]