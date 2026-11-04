"""
hand_feedback.py — did the hand get the object? From the hand's own sensors, the same way on the real Ability Hand
(taco_bandit smart_hand_node: 30 FSR touch values + 6 motor positions) and in MuJoCo (the model's 30 touch sensors on the same FSR
sites, the finger joints):

    from steve_grasping.hand_feedback import TouchConfig, grasp_state
    st = grasp_state(touch30, closure, cmd_closure, TouchConfig.from_tool(tool, "real"))
    st.state        # open | held | contact | blocked | empty
    st.holding      # something is in the hand (held / contact / blocked)
    st.fingers      # per finger in contact (index, middle, ring, pinky, thumb)

  held     the thumb and at least min_fingers - 1 fingers touch something: an object between thumb and fingers
  contact  something touches, but not thumb-against-fingers (the object pushed aside, a finger on the table)
  blocked  no fingertip touch, the fingers stopped short of closed: something in the palm — a wrap holds a ball with
           the inner finger segments, where the hand has no FSRs (while still closing it reads the same)
  empty    the fingers closed (nearly) all the way without touching anything: nothing in the hand
  open     commanded open

The touch array is the hand's order: 6 FSRs per finger, fingers [index, middle, ring, pinky, thumb] (smart_hand_node
/hand/tactile_frame touch_raw; the MuJoCo sites abh_fsr0..29). Per finger the 6 values are summed, so the
order within a finger does not matter. On the real hand the values are tared first (TouchTare, tool config
touch.real.tare): the threshold applies to the rise over each sensor's resting value. Thresholds per source
(tool config `touch:` sim / real): MuJoCo touch sensors measure newtons, the real FSRs ah_wrapper's newton
estimate — check with `ros2 topic echo /hand/tactile_frame --field touch_raw` while squeezing a finger.
"""

from dataclasses import dataclass

import numpy as np

FINGERS = ("index", "middle", "ring", "pinky", "thumb")
STATES = ("open", "held", "contact", "blocked", "empty")


@dataclass
class TouchConfig:
    threshold: float = 0.3       # per finger (sum of its 6 FSRs): above = contact
    min_fingers: int = 2         # fingers in contact for "held", the thumb among them
    closed_frac: float = 0.92    # closure >= this * commanded closure: closed through (nothing in between)
    open_below: float = 0.3      # commanded closure below this: "open"
    tare: bool = False           # compare the rise over each FSR's resting value (TouchTare), not the raw value
    tare_alpha: float = 0.05     # resting-value update per sample while the hand rests open
    tare_cap: float = 0.15       # per FSR: a rise above rest larger than this while open is a touch, not learnt

    @classmethod
    def from_tool(cls, tool, source):
        """`touch:` of the tool config: shared keys, overridden per source (sim / real)."""
        c = dict((tool.config or {}).get("touch") or {})
        per = c.pop("sim", {}) or {}, c.pop("real", {}) or {}
        c.update(per[0] if source == "sim" else per[1])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in c.items() if k in known})


@dataclass
class GraspState:
    state: str
    fingers: np.ndarray          # bool per finger
    touch: np.ndarray            # per-finger sums
    closure: float
    cmd_closure: float

    @property
    def n_fingers(self):
        return int(np.sum(self.fingers))

    @property
    def holding(self):
        """Something is in the hand: held by thumb and fingers, touched, or blocking the fingers."""
        return self.state in ("held", "contact", "blocked")

    def __str__(self):
        on = [f for f, c in zip(FINGERS, self.fingers) if c]
        return (f"{self.state} (closure {self.closure:.2f} of {self.cmd_closure:.2f}, touch: "
                f"{', '.join(on) if on else 'none'})")

    def as_dict(self):
        return dict(state=self.state, fingers=[bool(f) for f in self.fingers],
                    touch=[float(t) for t in self.touch], closure=float(self.closure),
                    cmd_closure=float(self.cmd_closure))

    @classmethod
    def from_dict(cls, d):
        return cls(d["state"], np.asarray(d["fingers"], bool), np.asarray(d["touch"], float),
                   float(d["closure"]), float(d["cmd_closure"]))


def finger_touch(touch):
    """Per-finger sums of the 30 FSR values (fewer values: zero-padded)."""
    t = np.zeros(30)
    v = np.asarray(touch if touch is not None else [], float).ravel()[:30]
    t[:len(v)] = np.nan_to_num(v)
    return t.reshape(5, 6).sum(axis=1)


class TouchTare:
    """Resting value of each of the 30 FSRs, to compare the rise over it with the threshold.

    The real FSRs do not read 0 at rest: ah_wrapper's conversion adds C_2 = 0.88 N to every non-zero raw value
    (about 7 per finger summed), and a sensor under preload (the thumb's) rests higher still — so a raw threshold
    reads contact all the time. The resting value is tracked while the hand is commanded open and its fingers
    have stopped (update(raw, resting=True): low-pass, tare_alpha per sample, only on sensors within cap of their
    resting value — a fingertip on the table or the object while open is not learnt as rest) and always lowered to
    a smaller raw value (a sensor that relaxes, or rest before the first open: the minimum seen so far)."""

    def __init__(self, alpha=0.05, cap=0.15):
        self.alpha, self.cap, self.base, self.n_rest = float(alpha), float(cap), None, 0

    def update(self, raw, resting):
        v = np.zeros(30)
        r = np.nan_to_num(np.asarray(raw if raw is not None else [], float).ravel()[:30])
        v[:len(r)] = r
        if self.base is None:
            self.base = v.copy()
        elif resting:
            near = v - self.base < self.cap
            self.base[near] += self.alpha * (v[near] - self.base[near])
        if resting:
            self.n_rest += 1
        self.base = np.minimum(self.base, v)

    def apply(self, raw):
        """The rise of each FSR over its resting value (>= 0); raw unchanged before the first sample."""
        if raw is None or self.base is None:
            return raw
        v = np.zeros(30)
        r = np.nan_to_num(np.asarray(raw, float).ravel()[:30])
        v[:len(r)] = r
        return np.maximum(v - self.base, 0.0)


def closure_of(pos, open_, close):
    """Closure 0..1 of hand positions between two synergies (mean over the motors that differ)."""
    pos, open_, close = (np.asarray(x, float) for x in (pos, open_, close))
    d = close - open_
    m = np.abs(d) > 1e-6
    return float(np.clip(np.mean((pos[m] - open_[m]) / d[m]), 0.0, 1.0)) if m.any() else 0.0


def grasp_state(touch, closure, cmd_closure, cfg=TouchConfig()):
    per = finger_touch(touch)
    fingers = per > cfg.threshold
    n, thumb = int(fingers.sum()), bool(fingers[4])
    if cmd_closure < cfg.open_below:
        s = "open"
    elif thumb and n >= cfg.min_fingers:
        s = "held"
    elif n > 0:
        s = "contact"
    elif closure >= cfg.closed_frac * cmd_closure:
        s = "empty"
    else:
        s = "blocked"
    return GraspState(s, fingers, per, float(closure), float(cmd_closure))
