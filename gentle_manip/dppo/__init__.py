"""DPPO (Diffusion Policy Policy Optimization) integration for the genesis sim.

Bridges DPPO (third_party/dppo, envs/dppo py3.10) to our genesis sim (envs/sim 3.12) over
the pure-python rpc socket, mirroring the SERL/DP3 bridges. Genesis-free (imports only
gentle_manip.envs.rpc + numpy), so it loads inside envs/dppo.
"""
