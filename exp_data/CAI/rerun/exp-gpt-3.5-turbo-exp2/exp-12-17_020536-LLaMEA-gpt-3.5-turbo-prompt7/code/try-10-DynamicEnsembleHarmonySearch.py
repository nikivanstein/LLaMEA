class DynamicEnsembleHarmonySearch(HarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        ensemble_size = 5  # Increase ensemble size for diversity
        for _ in range(self.budget):
            ensembles = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(ensemble_size)]
            for harmony in ensembles:
                for i in range(self.dim):
                    if np.random.rand() < pitch_rate:
                        harmony[i] = np.random.choice(self.harmony_memory[:, i])
                if func(harmony) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = harmony
                    self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
            pitch_rate = max(0.1, pitch_rate * 0.99 * 1.087)
            for i in range(self.dim):
                new_harmony_local = np.copy(self.harmony_memory[-1])
                new_harmony_local[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - 0.1),
                                                         min(self.upper_bound, self.harmony_memory[-1][i] + 0.1))
                if func(new_harmony_local) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = new_harmony_local
        return self.harmony_memory[0]