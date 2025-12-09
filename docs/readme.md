# This document has all the information and the motivation behind every step and equation used.

## The Three Steps That Repeat Every Timestep

┌─────────────────────────────────────────┐
│  SIMULATION LOOP (happens 500+ times)   │
└─────────────────────────────────────────┘
           │
           ▼
    STEP 1: COMPUTE FORCES
    ┌──────────────────────────────────────┐
    │ For each particle:                    │
    │   Look at ALL other particles        │
    │   Calculate: F = GMm/r²              │
    │   Sum up all forces on this particle │
    └──────────────────────────────────────┘
           │
           ▼
    STEP 2: UPDATE VELOCITIES
    ┌──────────────────────────────────────┐
    │ For each particle:                    │
    │   v_new = v_old + (F/m) × dt         │
    │   (Force / mass = acceleration)      │
    │   (acceleration × timestep = change) │
    └──────────────────────────────────────┘
           │
           ▼
    STEP 3: UPDATE POSITIONS
    ┌──────────────────────────────────────┐
    │ For each particle:                    │
    │   pos_new = pos_old + velocity × dt  │
    │   (Move based on current velocity)   │
    └──────────────────────────────────────┘
           │
           ▼
    [Go back to STEP 1 for next timestep]


Why This Works (The Math)
Newton's Second Law:

F = m × a
Therefore: a = F / m

This is why we divide force by mass!
Integration (Position Update):

position_new = position_old + velocity × timestep

Example:
  - Current position: x = 0 meters
  - Current velocity: v = 10 meters/second
  - Timestep: dt = 0.1 seconds
  - Next position: x = 0 + 10 × 0.1 = 1 meter


## Data Structure

Every particle must have 4 props:

    class Particle:
    position = [x, y, z]         # (3D vector)
    velocity = [vx, vy, vz]      # (3D vector)
    acceleration = [ax, ay, az]  # (3D vector)
    mass = m                     # (scalar)

Why Taichi? (GPU physics engine)

Regular python:
    # This is SLOW on CPU
for i in range(1_000_000):
    for j in range(1_000_000):
        calculate_force(i, j)  # 1 trillion operations!

Taichi:

    # Taichi parallelizes automatically!
@ti.kernel
def compute_forces():
    for i in range(n_particles):      # Can run 1000s simultaneously!
        for j in range(n_particles):
            calculate_force(i, j)


what Taichi does:

    Automatically parallelizes loops across CPU cores
    On ny i7 12th gen: 8 cores = ~8x speedup
    Can be moved to GPU later: ~100x more speedup
    Same code, automatic optimization

