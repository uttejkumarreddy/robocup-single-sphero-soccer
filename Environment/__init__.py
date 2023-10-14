from Configurations import Environment as envProps
from Environment.Simulation import Simulation
from Environment.Environment_1A_0D_0K import Environment_1A_0D_0K


def make(model):
    if model == "1A_0D_0K":
        env = Environment_1A_0D_0K()
    else:
        raise NotImplementedError

    simulation = Simulation(env)

    return simulation
