from . import mns5217, naive

BASELINES = {"mns5217": mns5217.build, "naive": naive.build}


def get_baseline(name: str):
    return BASELINES[name]()
