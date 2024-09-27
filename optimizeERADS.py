import os
import numpy as np
from ioh import get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA
from misc.utils import budget_logger
import time


from itertools import product
from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import Scenario
from smac import AlgorithmConfigurationFacade

from llamea.bbobalgs.ERADS_QuantumFluxUltraRefined import ERADS_QuantumFluxUltraRefined


def optimzeERADS( dim
):
    
    auc_mean = 0
    dim = dim
    budget = 2000 * dim
    error = ""
    algorithm = ERADS_QuantumFluxUltraRefined


    # perform a small run to check for any code errors
    l2_temp = aoc_logger(100, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = get_problem(11, 1, dim)
    problem.attach_logger(l2_temp)
    try:
        algorithm = ERADS_QuantumFluxUltraRefined(budget=100, dim=dim)
        algorithm(problem)
    except OverBudgetException:
        pass

    # now optimize the hyper-parameters
    def get_bbob_performance(config: Configuration, instance: str, seed: int = 0):
        np.random.seed(seed)
        fid, iid = instance.split(",")
        fid = int(fid[1:])
        iid = int(iid[:-1])
        problem = get_problem(fid, iid, dim)
        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        problem.attach_logger(l2)
        try:
            algorithm = ERADS_QuantumFluxUltraRefined(
                budget=budget, dim=dim, **dict(config)
            )
            algorithm(problem)
        except OverBudgetException:
            pass
        except Exception as e:
            print(problem.state, budget, e)
        auc = correct_aoc(problem, l2, budget)
        return 1 - auc

    args = list(product(range(1, 25), range(1, 4)))
    np.random.shuffle(args)
    inst_feats = {str(arg): [arg[0]] for idx, arg in enumerate(args)}
    # inst_feats = {str(arg): [idx] for idx, arg in enumerate(args)}
    error = ""

    configuration_space  = ConfigurationSpace({"population_size":(10,100), "F_init":(0.0, 1.0), "F_end":(0.0,1.0), "CR":(0.0, 1.0), "memory_factor":(0.0,1.0)})
    
    scenario = Scenario(
        configuration_space,
        name="ERADS",
        deterministic=False,
        min_budget=12,
        max_budget=200,
        n_trials=5000,
        instances=args,
        instance_features=inst_feats,
        output_directory=f"smac3_d{dim}"
        #n_workers=10
    )
    smac = AlgorithmConfigurationFacade(scenario, get_bbob_performance) #, logging_level=30
    incumbent = smac.optimize()

    # last but not least, perform the final validation
    
    loggers = [budget_logger(budget=budget, triggers=[logger.trigger.ALWAYS]), logger.Analyzer(folder_name=f"ioh/ERADS{dim}", algorithm_name="ERADS")]
    l1 = logger.Combine(loggers)
    aucs = []
    for fid in np.arange(1, 25):
        for iid in [1, 2, 3 , 4, 5]:
            problem = get_problem(fid, iid, dim)
            problem.attach_logger(l1)
            for rep in range(3):
                np.random.seed(rep)
                try:
                    algorithm = ERADS_QuantumFluxUltraRefined(budget=budget, dim=dim, **dict(incumbent))
                    algorithm(problem)
                except OverBudgetException:
                    pass
                for l in loggers:
                    l.reset()
                problem.reset()

    dict_hyperparams = dict(incumbent)
    print(dict_hyperparams)
    feedback = f"The algorithm ERADS got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with optimal hyperparameters {dict_hyperparams}."
    print("ERADS", algorithm)
    
    return


for dim in [10,20]:
    optimzeERADS(dim)