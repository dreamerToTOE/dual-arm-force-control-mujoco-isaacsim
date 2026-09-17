# Task05 3D Translational Cartesian Impedance — Experiment Record (2026-09-17)

## Setup

- Frame: BASE
- External force pulse: `F_ext = [8, 0, 0] N`, `t in [1.0, 3.0) s`
- Controller: `F_cmd = Kx (p_des-p) + Dx (v_des-v)`
- Torque mapping: `tau_task = Jv^T F_cmd`
- Gravity compensation: enabled
- Orientation control: intentionally disabled in this first 3D stage

## Results

| case | Kx [N/m] | Dx [N s/m] | peak position error [m] | pulse-end |dx| [m] | ideal F/K [m] | recovery <5 mm | max orientation drift [deg] |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft | 120 | 28 | 0.03331 | 0.03299 | 0.06667 | not reached | 9.788 |
| hard | 400 | 48 | 0.01177 | 0.01164 | 0.02000 | not reached | 3.494 |
| low_damping | 250 | 5 | 0.02451 | 0.02433 | 0.03200 | not reached | 6.995 |
| damped | 250 | 35 | 0.01815 | 0.01799 | 0.03200 | not reached | 5.441 |

## Interpretation

1. Stiffness trend is correct: under the same +8 N disturbance, the hard Cartesian spring produces much less TCP displacement than the soft spring.
2. Damping trend is correct: low damping produces a larger velocity peak and a stronger rebound after force removal; larger damping suppresses the transient but returns more slowly.
3. The simple static estimate `dx = F/Kx` is only an ideal spring relation. The simulated FR3 also contains passive joint damping/frictionloss and the 3D controller leaves orientation/uncontrolled null-space motion free, so the full equilibrium is not described by `F_ext + Kx dx = 0` alone.
4. All `recovery < 5 mm` metrics are `nan`: the TCP moves back toward the target after the pulse, but within the simulated recovery window it does not enter and stay inside the 5 mm ball. Residual passive/friction effects are a likely contributor; this must not be reported as a complete recovery pass.
5. Orientation drift is expected because this stage controls only TCP translation. It is not a controller bug; 6D Cartesian impedance will explicitly regulate orientation.

## Status

- Core 3D impedance concepts: PASS.
- Numerical stability: PASS (no NaN/Inf).
- Stiffness comparison: PASS.
- Damping comparison: PASS.
- Strict 5 mm recovery criterion: NOT YET PASSED.
- 6D position + orientation impedance: NOT YET IMPLEMENTED.
