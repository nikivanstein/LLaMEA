import os
from datetime import datetime

import numpy as np


class ExperimentLogger:
    def __init__(self, name=""):
        """
        Initializes an instance of the ExperimentLogger.
        Sets up a new logging directory named with the current date and time.

        Args:
            name (str): The name of the experiment.
        """
        self.dirname = self.create_log_dir(name)

    def create_log_dir(self, name=""):
        """
        Creates a new directory for logging experiments based on the current date and time.
        Also creates subdirectories for IOH experimenter data and code files.

        Returns:
            str: The name of the created directory.
        """
        today = datetime.today().strftime("%m-%d_%H%M%S")
        dirname = f"exp-{today}-{name}"
        os.mkdir(dirname)
        os.mkdir(f"{dirname}/ioh")
        os.mkdir(f"{dirname}/code")
        return dirname

    def log_conversation(self, content):
        """
        Logs the given conversation content into a conversation log file.

        Args:
            content (str): The conversation content to be logged.
        """
        with open(f"{self.dirname}/conversationlog.txt", "a") as file:
            file.write(f"\n\n-- {datetime.now()}\n")
            file.write(content)

    def log_code(self, attempt, algorithm_name, code):
        """
        Logs the provided code into a file, uniquely named based on the attempt number and algorithm name.

        Args:
            attempt (int): The attempt number of the code execution.
            algorithm_name (str): The name of the algorithm used.
            code (str): The source code to be logged.
        """
        with open(
            f"{self.dirname}/code/try-{attempt}-{algorithm_name}.py", "w"
        ) as file:
            file.write(code)

    def log_aucs(self, attempt, aucs):
        """
        Logs the given AOCCs (Area Over the Convergence Curve, named here auc) into a file, named based on the attempt number.

        Args:
            attempt (int): The attempt number corresponding to the AOCCs.
            aucs (array_like): An array of AUC scores to be saved.
        """
        np.savetxt(f"{self.dirname}/try-{attempt}-aucs.txt", aucs)
