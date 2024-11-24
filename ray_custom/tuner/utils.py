from typing import Dict


def execute_for_struct(a: dict, b: dict, leaf_fun):
    if isinstance(b, dict) and isinstance(a, dict):
        return any(k in a and execute_for_struct(a[k], b[k], leaf_fun) for k in b)
    return leaf_fun(a, b)


def get_tune_trial_stopper(stop_conditions):
    def tune_trial_stopper(trial_id: str, result: Dict):
        nonlocal stop_conditions
        is_result_greater_than_stop_condition = lambda result_val, stop_condition_val: result_val > stop_condition_val
        if execute_for_struct(result, stop_conditions, is_result_greater_than_stop_condition):
            return True
        return False

    return tune_trial_stopper
