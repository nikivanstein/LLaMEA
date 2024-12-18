class FastDynamicBandwidthHarmonySearch(HarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.7
        bandwidth = 0.1
        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for i in range(self.dim):
                if np.random.rand() < pitch_rate:
                    new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
            if func(new_harmony) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmony
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
            if _ % 10 == 0:
                pitch_rate = max(0.01, pitch_rate * 0.97 * 1.3)
                bandwidth = min(0.5, max(0.01, bandwidth * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))))
        return self.harmony_memory[0]