class ImprovedHarmonySearch(HarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        prev_best_solution = np.copy(self.harmony_memory[0])
        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for i in range(self.dim):
                if np.random.rand() < pitch_rate:
                    new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
            if func(new_harmony) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmony
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
            if func(self.harmony_memory[0]) < func(prev_best_solution):
                prev_best_solution = np.copy(self.harmony_memory[0])
                pitch_rate = max(0.1, pitch_rate * 0.99 * 1.087)  # Increased pitch adjustment rate for faster exploration
            for i in range(self.dim):
                new_harmony_local = np.copy(prev_best_solution)
                new_harmony_local[i] = np.random.uniform(max(self.lower_bound, prev_best_solution[i] - 0.1),
                                                         min(self.upper_bound, prev_best_solution[i] + 0.1))
                if func(new_harmony_local) < func(prev_best_solution):
                    prev_best_solution = np.copy(new_harmony_local)
        return prev_best_solution