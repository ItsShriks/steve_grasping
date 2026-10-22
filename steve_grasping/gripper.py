"""
gripper.py — open / close the tool's gripper in MuJoCo, from the tool config (config/tools/<tool>.yaml).

    from steve_grasping.gripper import SimGripper
    grip = SimGripper(model, load_tool("ability_hand"))
    grip.command(data, close=True)          # or grip.command(data, fraction=0.5): between open and close

Two kinds of tools:

  single actuator  sim: {actuator: fingers_actuator, open: 0, close: 255}           (Robotiq 2F-85)
  hand synergies   synergies: {open: ..., close: ...} (top level, shared with the real hand) or
                   sim: {synergies: ...}; a synergy is a list in the order of sim.actuators, or a dict
                   {actuator (without sim.actuator_prefix): target}; sim.units: rad (default) | deg
                   (Ability Hand: the API's degrees -> the MuJoCo position actuators' radians)
"""

import numpy as np


class SimGripper:
    def __init__(self, model, tool):
        import mujoco
        self.ids, self.open, self.close = [], np.zeros(0), np.zeros(0)
        if not tool.gripper:
            return
        sim = tool.sim
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if sim.get("actuator"):
            self.ids = [aid(sim["actuator"])]
            self.open, self.close = np.array([sim.get("open", 0.0)]), np.array([sim.get("close", 255.0)])
        else:
            syn = sim.get("synergies") or tool.config.get("synergies") or {}
            if "open" not in syn or "close" not in syn:
                raise ValueError(f"tool {tool.name}: sim needs `actuator` or `synergies` with open / close")
            prefix, scale = sim.get("actuator_prefix", ""), np.radians(1.0) if sim.get("units") == "deg" else 1.0
            names = sim.get("actuators")
            if isinstance(syn["open"], dict):
                names = list(dict.fromkeys([*syn["open"], *syn["close"]]))
                pick = lambda s: [s.get(n, syn["open"].get(n, 0.0)) for n in names]
                o, c = pick(syn["open"]), pick(syn["close"])
            else:
                o, c = syn["open"], syn["close"]
                if names is None or len(names) != len(o) or len(names) != len(c):
                    raise ValueError(f"tool {tool.name}: list synergies need sim.actuators of the same length")
            self.ids = [aid(prefix + n) for n in names]
            self.open, self.close = np.asarray(o, float) * scale, np.asarray(c, float) * scale
        missing = [i for i, a in enumerate(self.ids) if a < 0]
        if missing:
            raise ValueError(f"tool {tool.name}: gripper actuators not in the model: {missing}")

    def __bool__(self):
        return bool(self.ids)

    def targets(self, close=True, fraction=None):
        f = (1.0 if close else 0.0) if fraction is None else float(np.clip(fraction, 0.0, 1.0))
        return self.open + f * (self.close - self.open)

    def command(self, data, close=True, fraction=None):
        if self.ids:
            data.ctrl[self.ids] = self.targets(close, fraction)
