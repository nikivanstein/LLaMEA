import numpy as np

class NovelFastConvergingGreyWolfOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def get_alpha_beta_delta(wolves):
            sorted_wolves = sorted(wolves, key=lambda x: x['fitness'])
            return sorted_wolves[0], sorted_wolves[1], sorted_wolves[2]

        def update_position(wolf, alpha, beta, delta, prev_fitness, diversity):
            fitness_improvement = prev_fitness - wolf['fitness']
            step_size = 1.0 - np.exp(-0.1 * fitness_improvement)  # Dynamic step size based on fitness improvement
            a = 1.8 - 1.6 * (np.arange(self.dim) / (self.dim - 1)) * step_size

            # Adaptive mutation based on population diversity
            mutation_rate = 0.1 + 0.4 * diversity
            r = np.random.normal(0, mutation_rate, self.dim)

            new_position = np.clip(wolf['position'] + r, -5.0, 5.0)
            return new_position

        wolves = [{'position': np.random.uniform(-5.0, 5.0, self.dim),
                   'fitness': func(np.random.uniform(-5.0, 5.0, self.dim))} for _ in range(5)]

        for i in range(2, self.budget - 3, 2):  # Dynamic population size adaptation
            alpha, beta, delta = get_alpha_beta_delta(wolves)
            diversity = np.mean([np.linalg.norm(w1['position'] - w2['position']) for idx, w1 in enumerate(wolves) for w2 in wolves[idx + 1:]])
            for wolf in wolves:
                prev_fitness = wolf['fitness']
                wolf['position'] = update_position(wolf, alpha, beta, delta, prev_fitness, diversity)
                wolf['fitness'] = func(wolf['position'])
            if i < self.budget - 3:
                wolves.append({'position': np.random.uniform(-5.0, 5.0, self.dim),
                               'fitness': func(np.random.uniform(-5.0, 5.0, self.dim))})

        best_wolf = min(wolves, key=lambda x: x['fitness'])
        return best_wolf['position']