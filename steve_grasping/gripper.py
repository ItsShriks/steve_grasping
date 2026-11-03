"""
gripper.py — open / close the tool's gripper in MuJoCo, from the tool config (config/tools/<tool>.yaml).

    from steve_grasping.gripper import SimGripper
    grip = SimGripper(model, load_tool("ability_hand"))
    grip.command(data, close=True)          # or grip.command(data, fraction=0.5): between open and close
    grip.set_type("pinch")                  # a hand's grasp type (tool config grasp_types): its synergy
    grip.touch(data), grip.closure(data)    # 30 FSR touch values [N], finger closure 0..1 (hand_feedback)

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
        self.type, self.cmd, self.qadr, self.touch_adr = None, 0.0, [], None
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
            # grasp types with their own synergy (list form, same actuators)
            self.types = {None: (self.open, self.close)}
            for name, g in ((tool.config or {}).get("grasp_types") or {}).items():
                sg = (g or {}).get("synergy")
                self.types[name] = ((np.asarray(sg["open"], float) * scale, np.asarray(sg["close"], float) * scale)
                                    if sg else (self.open, self.close))
        missing = [i for i, a in enumerate(self.ids) if a < 0]
        if missing:
            raise ValueError(f"tool {tool.name}: gripper actuators not in the model: {missing}")
        self.type, self.cmd = None, 0.0
        joint = all(model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT for a in self.ids)
        self.qadr = [model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in self.ids] if joint else []
        # touch sensors on the hand's FSR sites abh_fsr0..29 (the real hand's order), if the model has them
        site = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, model.sensor_objid[i]): model.sensor_adr[i]
                for i in range(model.nsensor) if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH}
        adr = [site.get(f"abh_fsr{k}") for k in range(30)]
        self.touch_adr = adr if all(a is not None for a in adr) else None

    def __bool__(self):
        return bool(self.ids)

    def set_type(self, name):
        """Grasp type (tool config grasp_types) whose synergy the next command uses; unknown: the default."""
        types = getattr(self, "types", {None: (self.open, self.close)})
        self.type = name if name in types else None
        self.open, self.close = types[self.type]

    def targets(self, close=True, fraction=None):
        f = (1.0 if close else 0.0) if fraction is None else float(np.clip(fraction, 0.0, 1.0))
        self.cmd = f
        return self.open + f * (self.close - self.open)

    def command(self, data, close=True, fraction=None):
        if self.ids:
            data.ctrl[self.ids] = self.targets(close, fraction)

    def closure(self, data):
        """Finger closure 0..1 between the current type's open and close synergy (joint positions)."""
        from steve_grasping.hand_feedback import closure_of
        return closure_of(data.qpos[self.qadr], self.open, self.close) if self.qadr else float(self.cmd)

    def touch(self, data):
        """The 30 FSR touch values [N] (None: the model has no touch sensors)."""
        return None if self.touch_adr is None else np.array([data.sensordata[a] for a in self.touch_adr])
