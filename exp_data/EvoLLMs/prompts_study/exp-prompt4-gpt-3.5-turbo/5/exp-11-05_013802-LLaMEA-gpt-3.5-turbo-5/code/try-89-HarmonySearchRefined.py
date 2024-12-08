import numpy as np

class HarmonySearchRefined(HarmonySearch):
    def __call__(self, func):
        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            pitch_adjust_rate = 1 - 0.9 * _ / self.budget  # Dynamic pitch adjustment rate
            for i in range(self.dim):
                if np.random.rand() < pitch_adjust_rate:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_solution[i] = self.harmony_memory[idx, i]
                else:
                    dynamic_bandwidth = self.bandwidth * (1 - _ / self.budget)  # Dynamic bandwidth adjustment
                    if np.random.rand() < 0.9:  # Introduce Levy flight mutation with 90% probability
                        step = np.random.standard_cauchy() / np.sqrt(np.abs(np.random.randn()))
                        new_solution[i] = self.harmony_memory[-1, i] + step
                    else:
                        best_solution = self.harmony_memory[0, i]
                        perturbation = np.random.normal(best_solution, dynamic_bandwidth / 2)  # Gaussian perturbation around the best solution
                        new_solution[i] = np.clip(perturbation, -5.0, 5.0)

            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
                self.harmony_memory = self.harmony_memory[np.argsort([func(sol) for sol in self.harmony_memory])]

        return self.harmony_memory[0]