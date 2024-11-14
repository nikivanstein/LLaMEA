import numpy as np

class FractalInspiredSelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.F_base = 0.5
        self.CR_base = 0.9

    def __call__(self, func):
        # Initialize best position using fractal-based initialization
        self.best_position = self.fractal_initialization(func)
        
        while self.func_evaluations < self.budget:
            new_population = np.copy(self.population)
            
            for i in range(self.population_size):
                # Fractal-inspired mutation: DE/rand/1 with fractal perturbation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                fractal_scale = np.cos(np.pi * self.func_evaluations / self.budget)
                mutant_vector = x1 + self.F_base * fractal_scale * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.copy(self.population[i])
                crossover_mask = np.random.rand(self.dim) < self.CR_base
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]

                # Selection
                trial_score = func(trial_vector)
                self.func_evaluations += 1

                if trial_score < func(self.population[i]):
                    new_population[i] = trial_vector
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_position = trial_vector

            self.population = new_population

            # Self-adaptive adjustment of F and CR
            self.F_base = 0.4 + 0.1 * np.random.rand()
            self.CR_base = 0.8 + 0.1 * np.random.rand()

        return self.best_position

    def fractal_initialization(self, func):
        fractal_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fractal_amplitude = (self.upper_bound - self.lower_bound) / 2.0
        fractal_phase = np.random.uniform(0, 4 * np.pi, (self.population_size, self.dim))
        fractal_function = fractal_positions + fractal_amplitude * np.cos(fractal_phase)

        best_fractal_score = float('inf')
        best_fractal_position = None

        for pos in fractal_function:
            score = func(pos)
            if score < best_fractal_score:
                best_fractal_score = score
                best_fractal_position = pos

        self.func_evaluations += self.population_size
        return best_fractal_position