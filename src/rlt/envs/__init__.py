from rlt.envs.base import BaseEnv

__all__ = ["BaseEnv"]

# Lazy imports to avoid requiring gym-aloha for non-ALOHA usage.
def __getattr__(name):
    if name == "AlohaSimEnv":
        from rlt.envs.aloha_env import AlohaSimEnv
        return AlohaSimEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
