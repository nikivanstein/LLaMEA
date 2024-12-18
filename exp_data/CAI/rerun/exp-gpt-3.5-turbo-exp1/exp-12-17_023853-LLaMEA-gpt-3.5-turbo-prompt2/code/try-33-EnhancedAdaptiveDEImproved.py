import numpy as np

class EnhancedAdaptiveDEImproved(AdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def generate_population():
            return np.random.uniform(-5.0, 5.0, size=(self.NP, self.dim))

        population = generate_population()
        best_solution = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.budget):
            trial_population = []
            for i in range(self.NP):
                idxs = [idx for idx in range(self.NP) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                strategy = np.random.choice([0, 1, 2, 3, 4])  # Added new mutation strategy

                F = self.F * np.exp(-_ / self.budget)  # Dynamic adjustment of F
                CR = self.CR * np.exp(-_ / self.budget)  # Dynamic adjustment of CR
                
                # New mechanism for self-adaptive control of mutation step sizes based on individual performance
                step_size = 0.1 + 0.9 * np.random.rand() if func(population[i]) <= func(best_solution) else 0.1 + 0.1 * np.random.rand()
                mutant = population[a] + step_size * (population[b] - population[c])
                
                j_rand = np.random.randint(self.dim)
                trial_ind = np.array([mutant[j] if np.random.rand() < CR or j == j_rand else population[i, j] for j in range(self.dim)])

                if func(trial_ind) < func(population[i]):
                    trial_population.append(trial_ind)
                else:
                    trial_population.append(population[i])

            population = np.array(trial_population)
            best_solution = population[np.argmin([func(ind) for ind in population])]

        return best_solution