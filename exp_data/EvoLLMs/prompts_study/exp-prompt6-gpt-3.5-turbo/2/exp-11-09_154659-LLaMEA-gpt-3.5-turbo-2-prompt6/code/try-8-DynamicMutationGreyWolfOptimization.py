import numpy as np

class DynamicMutationGreyWolfOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_mutation_rate = 0.1

    def __call__(self, func):
        def get_alpha_beta_delta(wolves):
            sorted_wolves = sorted(wolves, key=lambda x: x['fitness'])
            return sorted_wolves[0], sorted_wolves[1], sorted_wolves[2]

        def update_position(wolf, alpha, beta, delta, mutation_rate):
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            A1 = 2 * mutation_rate * r1 - mutation_rate
            C1 = 2 * r2

            D_alpha = np.abs(C1 * alpha['position'] - wolf['position'])
            X1 = alpha['position'] - A1 * D_alpha

            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            A2 = 2 * mutation_rate * r1 - mutation_rate
            C2 = 2 * r2

            D_beta = np.abs(C2 * beta['position'] - wolf['position'])
            X2 = beta['position'] - A2 * D_beta

            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            A3 = 2 * mutation_rate * r1 - mutation_rate
            C3 = 2 * r2

            D_delta = np.abs(C3 * delta['position'] - wolf['position'])
            X3 = delta['position'] - A3 * D_delta

            new_position = (X1 + X2 + X3) / 3
            new_position = np.clip(new_position, -5.0, 5.0)
            return new_position

        wolves = [{'position': np.random.uniform(-5.0, 5.0, self.dim),
                   'fitness': func(np.random.uniform(-5.0, 5.0, self.dim))} for _ in range(5)]

        mutation_rate = self.initial_mutation_rate

        for i in range(2, self.budget - 3, 2):  # Dynamic population size adaptation
            alpha, beta, delta = get_alpha_beta_delta(wolves)
            for wolf in wolves:
                wolf['position'] = update_position(wolf, alpha, beta, delta, mutation_rate)
                wolf['fitness'] = func(wolf['position'])
            if i < self.budget - 3:
                wolves.append({'position': np.random.uniform(-5.0, 5.0, self.dim),
                               'fitness': func(np.random.uniform(-5.0, 5.0, self.dim))})

            # Dynamic mutation rate adaptation based on individual performance
            best_wolf = min(wolves, key=lambda x: x['fitness'])
            worst_wolf = max(wolves, key=lambda x: x['fitness'])
            mutation_rate = self.initial_mutation_rate + 0.8 * (worst_wolf['fitness'] - best_wolf['fitness']) / self.budget

        best_wolf = min(wolves, key=lambda x: x['fitness'])
        return best_wolf['position']