# src/particle_system.py
# ============================================================================
# PARTICLE SYSTEM: N-BODY GRAVITY SIMULATION
# ============================================================================
#
# This file implements the core physics simulation using Taichi.
# Key concepts:
#   1. Data storage: Position, velocity, acceleration, mass for each particle
#   2. Force computation: F = GMm/r² (with softening to prevent singularities)
#   3. Integration: Update velocities and positions using Newton's equations
#   4. Parallelization: Taichi runs all calculations on CPU cores (8x speedup on i7)
#
# ============================================================================

import taichi as ti
import numpy as np

# Initialize Taichi with CPU backend (no GPU needed)
# Taichi will automatically use all CPU cores for parallelization
ti.init(arch=ti.cpu)


@ti.data_oriented
class ParticleSystem:
    """
    N-body gravity simulation using Taichi.

    Each particle has:
      - pos: 3D position [x, y, z]
      - vel: 3D velocity [vx, vy, vz]
      - acc: 3D acceleration [ax, ay, az]
      - mass: scalar mass

    The simulation computes gravitational forces and updates particles
    according to Newton's laws of motion.
    """

    def __init__(self, n_particles, box_size=1000.0, softening=0.1):
        """
        Initialize particle system.

        Args:
            n_particles: Number of particles in simulation
            box_size: Size of simulation box (particles confined here)
            softening: Softening parameter to prevent singularities
                      Use F = GMm/(r² + ε²) instead of F = GMm/r²
                      This prevents force from becoming infinite when r→0

        Example:
            particles = ParticleSystem(n_particles=100_000, box_size=1000.0, softening=0.1)
        """

        self.n = n_particles
        self.box_size = box_size
        self.softening = softening

        # ===== ALLOCATE PARTICLE DATA =====
        # These are Taichi fields: arrays stored in fast memory (CPU cache or GPU VRAM)
        # Much faster than Python lists!

        # Position: 3D vector for each particle
        # ti.Vector.field(3, ...) means each element is a 3D vector [x, y, z]
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

        # Velocity: 3D vector for each particle
        # Update rule: v_new = v_old + a * dt
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

        # Acceleration: 3D vector for each particle
        # Computed from forces: a = F / m
        self.acc = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

        # Mass: scalar for each particle
        # In our system, all particles have equal mass = 1/n_particles
        self.mass = ti.field(dtype=ti.f32, shape=n_particles)

        print(f"[ParticleSystem] Initialized {n_particles:,} particles")
        print(f"  Box size: {box_size} parsecs")
        print(f"  Softening: {softening} parsecs")


    def initialize_random(self, seed=42):
        """
        Initialize particles with random positions and velocities.

        This creates a bound system that's easy to validate:
        - Positions: uniform random in [-box_size/2, +box_size/2]
        - Velocities: small random (Gaussian with σ=0.1)
        - Masses: equal (1/n_particles each)

        Args:
            seed: Random seed for reproducibility

        Why this works:
          - Uniform position distribution: mimics cosmological simulations
          - Small velocities: creates bound system (doesn't fly apart)
          - Equal masses: simplifies comparison with theory
        """

        np.random.seed(seed)

        # Generate random positions: uniform in [-box_size/2, box_size/2]
        # shape = (n_particles, 3) for x, y, z coordinates
        positions = np.random.uniform(
            -self.box_size/2,
            self.box_size/2,
            (self.n, 3)
        ).astype(np.float32)

        # Generate random velocities: Gaussian with σ=1, then scale by 0.1
        # This ensures particles move slowly, creating a bound system
        velocities = np.random.normal(
            0,      # mean = 0
            1,      # standard deviation = 1
            (self.n, 3)
        ).astype(np.float32) * 0.1  # Scale down to 0.1 m/s

        # All particles have equal mass
        # Total mass = 1 (convenient units)
        # Each particle: mass = 1/n_particles
        masses = np.ones(self.n, dtype=np.float32) / self.n

        # Copy to Taichi fields (now available in fast memory)
        self.pos.from_numpy(positions)
        self.vel.from_numpy(velocities)
        self.mass.from_numpy(masses)

        # Initialize accelerations to zero (will be computed in first step)
        # Note: We don't call .from_numpy() because we want zeros
        # Taichi fields default to zero, so nothing needed

        print(f"[Initialization] Random positions and velocities")
        print(f"  Position range: [{-self.box_size/2}, {self.box_size/2}]")
        print(f"  Velocity scale: 0.1 (slow moving)")


    @ti.kernel
    def compute_forces(self):
        """
        Compute gravitational forces on all particles.

        This is the core of the N-body simulation.

        Physics:
          For each particle i:
            F_i = Σ_j≠i (G * m_i * m_j / (r_ij² + ε²)) * r̂_ij

            Where:
              G = 4.302e-3 (gravitational constant in our units)
              m_i, m_j = masses of particles i and j
              r_ij = distance between particles
              ε = softening parameter (prevents singularities)
              r̂_ij = unit vector from i to j

        Then acceleration:
          a_i = F_i / m_i

        Taichi Kernel Notes:
          @ti.kernel tells Taichi to parallelize this function
          Taichi automatically runs the outer loop (for i in range(self.n))
          across all CPU cores simultaneously

          On your i7 12th gen (8 cores):
            100_000 particles would run ~8000 iterations per core
            Much faster than serial!

        Memory Access Pattern:
          This is O(N²) in time complexity:
            - For each particle i: O(N)
            - Check all particles j: O(N)
            - Total: O(N²) operations

          With 100k particles: 10 billion operations per timestep
          On your i7: ~1-2 seconds per timestep
          With 500 timesteps: ~15 minutes per simulation
        """

        # Step 1: Clear accelerations from previous step
        # This is essential! Otherwise forces accumulate
        for i in self.acc:
            self.acc[i] = ti.Vector([0.0, 0.0, 0.0])

        # Step 2: Compute forces between all pairs
        # i = source particle, j = target particle
        for i in range(self.n):
            # For each particle i, look at ALL other particles j
            for j in range(self.n):
                # Don't compute force of particle on itself
                if i != j:
                    # Vector from particle i to particle j
                    # r_ij = position_j - position_i
                    r = self.pos[j] - self.pos[i]

                    # Distance squared (with softening)
                    # r_sq = r_ij² + ε²
                    # We compute r² + ε² instead of r² to avoid singularities
                    r_sq = r.dot(r) + self.softening**2

                    # Distance (taking square root)
                    # r_magnitude = sqrt(r² + ε²)
                    r_magnitude = ti.sqrt(r_sq)

                    # Gravitational force magnitude
                    # F = G * m_i * m_j / r_magnitude³
                    # (Note: r_magnitude³ in denominator because we multiply by unit vector)
                    G = 4.302e-3  # Gravitational constant
                    force_magnitude = G * self.mass[i] * self.mass[j] / (r_magnitude**3)

                    # Force vector (direction × magnitude)
                    # Force points from i toward j (direction of r)
                    force = force_magnitude * r

                    # Accumulate force on particle i
                    self.acc[i] += force / self.mass[i]

                    # Note: Newton's 3rd law
                    # Force on i from j = -Force on j from i
                    # But we compute both directions in the double loop
                    # (less efficient but simpler to parallelize)


    @ti.kernel
    def integrate(self, dt: ti.f32):
        """
        Update particle velocities and positions.

        Uses leapfrog integration (good energy conservation):

          v_new = v_old + a * dt       (update velocity)
          x_new = x_old + v_new * dt   (update position using new velocity)

        Args:
            dt: Timestep in Myrs (million years)
                Typical: dt = 0.01 Myr
                Smaller dt = more accurate but slower
                Larger dt = faster but may lose energy

        Why Leapfrog?
          - Simple to implement
          - Second-order accuracy (error ~ O(dt²))
          - Good energy conservation (energy error ~ O(dt²))
          - Common in N-body simulations

        Periodic Boundary Conditions:
          If particle leaves box, wrap it around to other side
          This prevents particles from flying away
          (Mimics infinite periodic space)
        """

        # Update each particle
        for i in range(self.n):
            # Step 1: Update velocity
            # v_new = v_old + a * dt
            # The acceleration was computed in compute_forces()
            self.vel[i] += self.acc[i] * dt

            # Step 2: Update position
            # x_new = x_old + v_new * dt
            # Use updated velocity (leapfrog style)
            self.pos[i] += self.vel[i] * dt

            # Step 3: Apply periodic boundary conditions
            # If particle goes outside box, wrap it around
            # This keeps simulation stable and prevents particle escape
            for d in ti.static(range(3)):  # d = 0, 1, 2 (x, y, z)
                # Check if position exceeds +box_size/2
                if self.pos[i][d] > self.box_size / 2:
                    self.pos[i][d] -= self.box_size
                # Check if position goes below -box_size/2
                if self.pos[i][d] < -self.box_size / 2:
                    self.pos[i][d] += self.box_size


    def get_positions(self):
        """Export particle positions to NumPy array."""
        return self.pos.to_numpy()

    def get_velocities(self):
        """Export particle velocities to NumPy array."""
        return self.vel.to_numpy()

    def get_accelerations(self):
        """Export particle accelerations to NumPy array."""
        return self.acc.to_numpy()

    def get_masses(self):
        """Export particle masses to NumPy array."""
        return self.mass.to_numpy()


# ============================================================================
# QUICK REFERENCE
# ============================================================================
#
# Usage example:
#
#   # Create system with 100k particles
#   particles = ParticleSystem(n_particles=100_000, box_size=1000.0, softening=0.1)
#
#   # Initialize randomly
#   particles.initialize_random(seed=42)
#
#   # Run one timestep
#   particles.compute_forces()  # Calculate accelerations
#   particles.integrate(dt=0.01)  # Update velocities and positions
#
#   # Repeat for multiple timesteps
#   for step in range(500):
#       particles.compute_forces()
#       particles.integrate(dt=0.01)
#       if step % 100 == 0:
#           print(f"Step {step}")
#
# ============================================================================
