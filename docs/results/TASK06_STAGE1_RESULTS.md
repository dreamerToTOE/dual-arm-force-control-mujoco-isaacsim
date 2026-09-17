# Task06 Stage 1 — First P vs PI Contact-Force Run

## Raw result

Target normal force:

```text
F_des = 10 N
```

Measured metrics:

```text
mode   contact[s]   rise90[s]   overshoot[%]   steady error[N]   peak force[N]   peak |tau|[Nm]
P         0.5600      0.0020         31.105            3.4313         13.1105          27.1835
PI        0.5600      0.0020         60.333            1.9057         16.0333          27.1835
```

## Interpretation

This run proves that the full contact-force feedback loop is operational:

```text
contact -> mj_contactForce -> force error -> P/PI -> J^T -> joint torque
```

The PI controller reduces the mean tail force error compared with P, but the current force trace shows severe high-frequency contact chatter. Therefore the run is a **numerical/execution PASS** but **not a control-quality PASS**.

The 2 ms `rise90` is dominated by the initial contact impulse and should not be interpreted as a meaningful closed-loop rise time.

The PI trace repeatedly approaches the command limits while measured contact force alternates strongly. A mean steady-state error alone is insufficient to judge contact-force quality because a rapidly oscillating force can still have a good average.

## Stage-1 status

Passed:

- real MuJoCo probe-plane contact established;
- `mj_contactForce` successfully measured the normal contact force;
- P and PI loops completed without NaN/Inf;
- PI reduced average tail error relative to P.

Not yet passed:

- stable contact without persistent high-frequency oscillation;
- acceptable force ripple and contact continuity;
- meaningful force rise/settling metrics after suppressing the initial impact transient.

## Next correction

Before moving to Task06 Stage 2, stabilize Stage 1 by:

- lowering force-loop gains;
- low-pass filtering the measured contact force used by the controller;
- limiting commanded push-force slew rate;
- reporting force ripple / standard deviation and contact-loss ratio in addition to mean steady-state error;
- only after the force loop is stable, tightening the contact model if penetration remains excessive.
