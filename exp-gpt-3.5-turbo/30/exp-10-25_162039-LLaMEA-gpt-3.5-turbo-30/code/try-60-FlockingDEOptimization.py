import numpy as np

class FlockingDEOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

    def _fitness(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def _update_position(self, population, best_position, w=0.5, c1=0.8, c2=0.9):
        r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
        velocity = w * population + c1 * r1 * (best_position - population) + c2 * r2 * (population - best_position)
        return np.clip(population + velocity, -5.0, 5.0)

    def _mutation(self, population, f=0.5, cr=0.7):
        mutant_population = np.zeros_like(population)
        for i in range(self.budget):
            idxs = [idx for idx in range(self.budget) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + f * (b - c), -5.0, 5.0)
            j_rand = np.random.randint(self.dim)
            trial = [mutant[j] if np.random.rand() < cr or j == j_rand else population[i, j] for j in range(self.dim)]
            mutant_population[i] = trial
        return mutant_population

    def __call__(self, func):
        population = self._initialize_population()
        best_position = population[np.argmin(self._fitness(population, func))]
        
        for _ in range(self.budget):
            population = self._update_position(population, best_position)
            population = self._mutation(population)
            current_best = population[np.argmin(self._fitness(population, func))]
            best_position = current_best if func(current_best) < func(best_position) else best_position
        
        return best_position