import numpy as np

class FineTunedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def generate_population():
            return np.random.uniform(-5.0, 5.0, size=(self.dim, self.dim))

        population = generate_population()
        best_solution = population[np.argmin([func(individual) for individual in population])]

        CR = self.CR_max
        F = self.F_max

        for _ in range(self.budget):
            new_population = []
            for idx, target in enumerate(population):
                a, b, c = np.random.choice([x for x in range(len(population)) if x != idx], size=3, replace=False)
                mutant = np.clip(population[a] + F * (population[b] - population[c]), -5.0, 5.0)
                trial = np.array([mutant[i] if np.random.uniform() < CR or i == np.random.randint(0, self.dim) else target[i] for i in range(self.dim)])
                if func(trial) < func(target):
                    population[idx] = trial
                    if func(trial) < func(best_solution):
                        best_solution = np.copy(trial)

            if np.random.uniform() < 0.1:
                CR = np.clip(CR + 0.05, self.CR_min, self.CR_max)
            if np.random.uniform() < 0.1:
                F = np.clip(F + 0.05, self.F_min, self.F_max)

            if np.random.uniform() < 0.3:  # Fine-tuning strategy
                # Modify individual lines with a 30% probability
                if np.random.uniform() < 0.3:
                    CR = np.random.uniform(self.CR_min, self.CR_max)
                if np.random.uniform() < 0.3:
                    F = np.random.uniform(self.F_min, self.F_max)

        return best_solution