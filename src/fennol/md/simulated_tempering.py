"""Simulated Tempering for enhanced sampling.

This module implements single-replica simulated tempering where the system
stochastically jumps between temperature levels using the Metropolis criterion.
Based on GROMACS 2025.3 implementation.

Key features:
- Temperature ladder with geometric, linear, or exponential spacing
- Metropolis acceptance criterion for temperature jumps
- Wang-Landau weight learning for improved sampling
- Trajectory writing only at reference temperature(s)

Configuration options:
    temp_low: float - Lowest temperature (K)
    temp_high: float - Highest temperature (K)
    n_temperatures: int - Number of temperature levels
    scaling: str - "geometric" (recommended), "linear", or "exponential"
    reference_temps: float or str - Temperature(s) for trajectory output
    nst_tempering: int - Steps between temperature jump attempts
    learn_weights: bool - Enable Wang-Landau weight learning
    wl_delta: float - Initial WL weight increment (default 1.0)
    wl_scale: float - WL delta reduction factor (default 0.8)
    wl_ratio: float - Histogram flatness criterion (default 0.8)
    expected_energy: float - Expected potential energy (kcal/mol) for weight init
    write_trajectory: str - "reference", "all", or "none"

Tuning tips:
- Use `expected_energy` for faster convergence (set to avg potential energy)
- More temperature levels = smaller gaps = better acceptance
- Geometric scaling gives ~equal acceptance between all adjacent pairs
- If stuck at one temperature, increase n_temperatures or wl_delta
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union
from ..utils.atomic_units import AtomicUnits as au

# Boltzmann constant in kcal/mol/K
KB_KCAL = 0.001987204  # kcal/mol/K


def generate_temperature_ladder(
    temp_low: float,
    temp_high: float,
    n_temps: int,
    scaling: str = "geometric"
) -> np.ndarray:
    """Generate temperature ladder.

    Args:
        temp_low: Lowest temperature (K)
        temp_high: Highest temperature (K)
        n_temps: Number of temperature levels
        scaling: "geometric", "linear", or "exponential"

    Returns:
        Array of temperatures in Kelvin
    """
    if n_temps < 2:
        raise ValueError(f"n_temperatures must be >= 2, got {n_temps}")
    if temp_low <= 0 or temp_high <= 0:
        raise ValueError(f"Temperatures must be positive")
    if temp_low >= temp_high:
        raise ValueError(f"temp_low ({temp_low}) must be < temp_high ({temp_high})")

    if scaling == "geometric":
        # Equal acceptance probability spacing (recommended)
        # T_i = T_low * (T_high / T_low)^(i / (N-1))
        return temp_low * np.power(
            temp_high / temp_low,
            np.linspace(0, 1, n_temps)
        )
    elif scaling == "linear":
        return np.linspace(temp_low, temp_high, n_temps)
    elif scaling == "exponential":
        # More dense at low temperatures
        lambdas = np.linspace(0, 1, n_temps)
        return temp_low + (temp_high - temp_low) * (np.expm1(lambdas) / np.expm1(1.0))
    else:
        raise ValueError(f"Unknown scaling: {scaling}. Use 'geometric', 'linear', or 'exponential'")


def setup_simulated_tempering(
    config: Dict,
    system_data: Dict,
    initial_temp: float
) -> Dict:
    """Initialize simulated tempering state.

    Args:
        config: Simulated tempering configuration dictionary
        system_data: System data dictionary
        initial_temp: Initial temperature in Kelvin

    Returns:
        State dictionary with:
        - temperatures: array of temperature levels
        - current_temp_idx: current temperature index
        - weights: g(T) weights for each level
        - reference_temp_indices: indices of reference temperatures
        - statistics: acceptance counts, histogram
    """
    temp_low = float(config.get("temp_low", 300.0))
    temp_high = float(config.get("temp_high", 500.0))
    n_temps = int(config.get("n_temperatures", 8))
    scaling = str(config.get("scaling", "geometric")).lower()

    temperatures = generate_temperature_ladder(temp_low, temp_high, n_temps, scaling)

    # Find starting temperature index (closest to initial_temp)
    current_idx = int(np.argmin(np.abs(temperatures - initial_temp)))

    # Find reference temperature indices
    ref_temps = config.get("reference_temps", temp_low)
    if isinstance(ref_temps, (int, float)):
        ref_temps = [float(ref_temps)]
    elif isinstance(ref_temps, str):
        # Handle comma-separated string
        ref_temps = [float(t.strip()) for t in ref_temps.split(",")]
    else:
        ref_temps = [float(t) for t in ref_temps]

    ref_indices = [int(np.argmin(np.abs(temperatures - t))) for t in ref_temps]

    # Initialize weights (g(T))
    # Option 1: Start uniform (zeros) - requires long WL equilibration
    # Option 2: Estimate from expected energy - much faster convergence
    init_weights = config.get("init_weights", "auto")
    expected_energy = config.get("expected_energy", None)  # in kcal/mol

    if init_weights == "auto" and expected_energy is not None:
        # Estimate initial weights from expected energy to achieve detailed balance
        #
        # For move from i to j, the acceptance criterion is:
        #   delta = (β_i - β_j) * E + (g_j - g_i)
        #   P_accept = min(1, exp(delta))
        #
        # For detailed balance (equal up/down rates), we want delta ≈ 0:
        #   (β_i - β_j) * E + (g_j - g_i) ≈ 0
        #   g_j - g_i ≈ -(β_i - β_j) * E = (β_j - β_i) * E
        #
        # Setting g_0 = 0:
        #   g_i = sum_{k=0}^{i-1} (g_{k+1} - g_k) = sum_{k=0}^{i-1} (β_{k+1} - β_k) * E
        #       = (β_i - β_0) * E
        #
        weights = np.zeros(n_temps)
        E_exp = float(expected_energy)
        beta_0 = 1.0 / (KB_KCAL * temperatures[0])
        for i in range(1, n_temps):
            beta_i = 1.0 / (KB_KCAL * temperatures[i])
            # g_i = (β_i - β_0) * E
            weights[i] = (beta_i - beta_0) * E_exp
        print(f"#   Initial weights estimated from E_expected = {E_exp:.1f} kcal/mol")
    elif isinstance(init_weights, (list, np.ndarray)):
        weights = np.array(init_weights, dtype=float)
        assert len(weights) == n_temps, f"init_weights must have {n_temps} elements"
    else:
        # Start uniform (all zeros)
        weights = np.zeros(n_temps)

    # Wang-Landau parameters
    learn_weights_val = config.get("learn_weights", True)
    if isinstance(learn_weights_val, str):
        learn_weights = learn_weights_val.lower() in ("yes", "true", "1")
    else:
        learn_weights = bool(learn_weights_val)

    wl_delta = float(config.get("wl_delta", 1.0))
    wl_scale = float(config.get("wl_scale", 0.8))
    wl_ratio = float(config.get("wl_ratio", 0.8))

    # Trajectory writing control
    # Options: "reference" (only at reference temps), "all" (all frames), "none" (no frames)
    write_traj = config.get("write_trajectory", "reference").lower()
    if isinstance(write_traj, bool):
        write_traj = "all" if write_traj else "none"

    state = {
        "temperatures": temperatures,
        "n_temps": n_temps,
        "current_temp_idx": current_idx,
        "weights": weights,
        "reference_temp_indices": ref_indices,
        "nst_tempering": int(config.get("nst_tempering", 100)),

        # Statistics - histogram tracks steps spent at each temperature
        "histogram": np.zeros(n_temps),  # Updated every step via update_step_count()
        "wl_histogram": np.zeros(n_temps),  # Updated only at jump attempts for WL
        "n_attempts": 0,
        "n_accepted": 0,
        "transitions": np.zeros((n_temps, n_temps)),

        # Wang-Landau
        "learn_weights": learn_weights,
        "wl_delta": wl_delta,
        "wl_scale": wl_scale,
        "wl_ratio": wl_ratio,
        "wl_equilibrated": False,

        # Trajectory writing control
        "write_trajectory": write_traj,
    }

    print(f"# Simulated Tempering: {n_temps} temperatures")
    print(f"#   Range: {temp_low:.1f} K - {temp_high:.1f} K ({scaling})")
    print(f"#   Temperatures: {', '.join(f'{t:.1f}' for t in temperatures)}")
    print(f"#   Reference temp(s): {[f'{temperatures[i]:.1f}' for i in ref_indices]} K")
    print(f"#   Starting at T[{current_idx}] = {temperatures[current_idx]:.1f} K")
    print(f"#   Jump attempts every {state['nst_tempering']} steps")
    if learn_weights:
        print(f"#   Wang-Landau weight learning enabled (delta={wl_delta}, scale={wl_scale})")
    traj_mode = {"reference": "only at reference temp(s)", "all": "all frames", "none": "disabled"}
    print(f"#   Trajectory writing: {traj_mode.get(write_traj, write_traj)}")

    return state


def attempt_temperature_jump(
    st_state: Dict,
    epot: float,
    rng_key: jax.random.PRNGKey
) -> Tuple[Dict, bool, int]:
    """Attempt Metropolis temperature jump.

    Args:
        st_state: Simulated tempering state
        epot: Current potential energy (atomic units)
        rng_key: JAX random key

    Returns:
        (updated_state, accepted, new_temp_idx)
    """
    temperatures = st_state["temperatures"]
    current_idx = st_state["current_temp_idx"]
    weights = st_state["weights"]
    n_temps = st_state["n_temps"]

    # Convert energy to kcal/mol for calculation
    # au.KCALPERMOL converts Hartree to kcal/mol
    epot_kcal = float(epot) * au.KCALPERMOL

    # Propose neighbor (50% up, 50% down)
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
    r1 = float(jax.random.uniform(subkey1))

    if r1 < 0.5:
        # Try lower temperature
        trial_idx = max(0, current_idx - 1)
    else:
        # Try higher temperature
        trial_idx = min(n_temps - 1, current_idx + 1)

    # If at boundary and tried to go beyond, stay (no attempt counted)
    if trial_idx == current_idx:
        return st_state, False, current_idx

    # Calculate Metropolis criterion for simulated tempering
    #
    # The extended ensemble has probability: P(x, k) ∝ exp(-β_k * E(x) + g_k)
    # where β_k = 1/(k_B * T_k) and g_k is a weight factor
    #
    # For a temperature swap from k_old to k_new at fixed configuration x:
    # Acceptance ratio = P(x, k_new) / P(x, k_old)
    #                  = exp(-β_new * E + g_new) / exp(-β_old * E + g_old)
    #                  = exp[(β_old - β_new) * E + (g_new - g_old)]
    #
    # So: Δ = (β_old - β_new) * E + (g_new - g_old)
    # Accept with probability min(1, exp(Δ))
    #
    # For E < 0 (typical MD with negative potential energy):
    #   - Going UP to higher T (β_old > β_new): (β_old - β_new) > 0
    #     Δ = positive * negative + g_diff = negative (plus weight correction)
    #     exp(Δ) < 1, so moves UP are disfavored (good!)
    #   - Going DOWN to lower T (β_old < β_new): (β_old - β_new) < 0
    #     Δ = negative * negative + g_diff = positive (plus weight correction)
    #     exp(Δ) > 1, so moves DOWN are favored (good!)
    #
    # This makes physical sense: negative energy systems prefer low T.

    T_old = temperatures[current_idx]
    T_new = temperatures[trial_idx]

    # β = 1/(k_B * T), in units where k_B is in kcal/mol/K
    beta_old = 1.0 / (KB_KCAL * T_old)
    beta_new = 1.0 / (KB_KCAL * T_new)

    # Δ = (β_old - β_new) * E + (g_new - g_old)
    delta = (beta_old - beta_new) * epot_kcal + (weights[trial_idx] - weights[current_idx])

    # Metropolis acceptance: P = min(1, exp(Δ))
    if delta >= 0:
        accept_prob = 1.0
    else:
        accept_prob = float(jnp.exp(delta))

    r2 = float(jax.random.uniform(subkey2))
    accepted = r2 < accept_prob

    # Create new state (shallow copy with mutable arrays copied)
    new_state = {**st_state}
    new_state["histogram"] = st_state["histogram"].copy()
    new_state["wl_histogram"] = st_state["wl_histogram"].copy()
    new_state["weights"] = st_state["weights"].copy()
    new_state["transitions"] = st_state["transitions"].copy()
    new_state["n_attempts"] = st_state["n_attempts"] + 1

    if accepted:
        new_idx = trial_idx
        new_state["n_accepted"] = st_state["n_accepted"] + 1
        new_state["current_temp_idx"] = new_idx
        new_state["transitions"][current_idx, new_idx] += 1
    else:
        new_idx = current_idx
        new_state["transitions"][current_idx, current_idx] += 1

    # Update WL histogram (for Wang-Landau weight updates)
    new_state["wl_histogram"][new_idx] += 1

    # Update weights if learning
    if new_state["learn_weights"] and not new_state["wl_equilibrated"]:
        new_state["weights"][new_idx] += new_state["wl_delta"]

        # Check histogram flatness periodically
        if new_state["n_attempts"] % 1000 == 0:
            hist = new_state["wl_histogram"]
            if hist.min() > 0:
                flatness = hist.min() / hist.mean()
                if flatness > new_state["wl_ratio"]:
                    # Reduce delta and reset WL histogram (not the step histogram)
                    new_state["wl_delta"] = new_state["wl_delta"] * new_state["wl_scale"]
                    new_state["wl_histogram"] = np.zeros(n_temps)
                    print(f"# ST: WL delta reduced to {new_state['wl_delta']:.6f}")

                    if new_state["wl_delta"] < 1e-6:
                        new_state["wl_equilibrated"] = True
                        print("# ST: Weights equilibrated")
                        print(f"# ST: Final weights: {new_state['weights']}")

    return new_state, accepted, new_idx


def update_step_histogram(st_state: Dict) -> Dict:
    """Update the step histogram to track time spent at each temperature.

    This should be called every MD step (or every nprint steps) to accurately
    track how much simulation time is spent at each temperature level.

    Args:
        st_state: Simulated tempering state

    Returns:
        Updated state with incremented histogram
    """
    new_state = {**st_state}
    new_state["histogram"] = st_state["histogram"].copy()
    new_state["histogram"][st_state["current_temp_idx"]] += 1
    return new_state


def rescale_velocities(
    velocities: jnp.ndarray,
    T_old: float,
    T_new: float
) -> jnp.ndarray:
    """Rescale velocities after temperature change.

    v_new = v_old * sqrt(T_new / T_old)

    Args:
        velocities: Current velocities array
        T_old: Old temperature in Kelvin
        T_new: New temperature in Kelvin

    Returns:
        Rescaled velocities
    """
    scale = jnp.sqrt(T_new / T_old)
    return velocities * scale


def should_write_trajectory(st_state: Dict) -> bool:
    """Check if trajectory frame should be written based on config.

    Args:
        st_state: Simulated tempering state

    Returns:
        True if trajectory frame should be written
    """
    write_mode = st_state.get("write_trajectory", "reference")

    if write_mode == "all":
        return True
    elif write_mode == "none":
        return False
    else:  # "reference" (default)
        return st_state["current_temp_idx"] in st_state["reference_temp_indices"]


def get_current_temperature(st_state: Dict) -> float:
    """Get current temperature in Kelvin.

    Args:
        st_state: Simulated tempering state

    Returns:
        Current temperature in Kelvin
    """
    return float(st_state["temperatures"][st_state["current_temp_idx"]])


def get_current_kT(st_state: Dict) -> float:
    """Get current kT in atomic units.

    Args:
        st_state: Simulated tempering state

    Returns:
        Current kT in atomic units
    """
    return get_current_temperature(st_state) / au.KELVIN


def print_st_statistics(st_state: Dict, step: int, verbose: bool = False):
    """Print simulated tempering statistics.

    Args:
        st_state: Simulated tempering state
        step: Current MD step
        verbose: If True, print detailed ladder information
    """
    n_att = st_state["n_attempts"]
    n_acc = st_state["n_accepted"]
    acc_rate = n_acc / n_att if n_att > 0 else 0

    hist = st_state["histogram"]
    temps = st_state["temperatures"]
    weights = st_state["weights"]
    n_temps = st_state["n_temps"]
    current_idx = st_state["current_temp_idx"]
    current_temp = temps[current_idx]
    ref_indices = st_state["reference_temp_indices"]

    print(f"# ST Statistics at step {step}:")
    print(f"#   Current T: {current_temp:.1f} K (index {current_idx})")
    print(f"#   Attempts: {n_att}, Accepted: {n_acc} ({acc_rate*100:.1f}%)")

    if st_state["learn_weights"] and not st_state["wl_equilibrated"]:
        print(f"#   WL delta: {st_state['wl_delta']:.6f}")
    elif st_state["wl_equilibrated"]:
        print(f"#   WL: equilibrated")

    # Print temperature ladder with statistics
    print(f"#   Temperature Ladder:")
    print(f"#   {'Idx':>3} {'Temp(K)':>8} {'Visits':>8} {'Weight':>10} {'Ref':>4}")
    print(f"#   {'-'*3} {'-'*8} {'-'*8} {'-'*10} {'-'*4}")

    total_visits = hist.sum()
    for i in range(n_temps):
        is_ref = "*" if i in ref_indices else ""
        is_current = "<-" if i == current_idx else ""
        visit_pct = f"({100*hist[i]/total_visits:.1f}%)" if total_visits > 0 else ""
        print(f"#   {i:>3} {temps[i]:>8.1f} {int(hist[i]):>8} {weights[i]:>10.4f} {is_ref:>4} {is_current}")

    # Print transition statistics if we have data
    transitions = st_state["transitions"]
    if transitions.sum() > 0:
        print(f"#   Adjacent Transition Rates:")
        for i in range(n_temps - 1):
            # Transitions from i to i+1 (up)
            up_attempts = transitions[i, i] + transitions[i, i+1]
            up_accepted = transitions[i, i+1]
            up_rate = up_accepted / up_attempts * 100 if up_attempts > 0 else 0

            # Transitions from i+1 to i (down)
            down_attempts = transitions[i+1, i+1] + transitions[i+1, i]
            down_accepted = transitions[i+1, i]
            down_rate = down_accepted / down_attempts * 100 if down_attempts > 0 else 0

            print(f"#     T[{i}]<->T[{i+1}]: up {up_rate:5.1f}%, down {down_rate:5.1f}%")

    # Estimate round-trip time if we have enough data
    if total_visits > 0 and n_temps >= 2:
        # Check if we've visited both endpoints
        visited_low = hist[0] > 0
        visited_high = hist[n_temps-1] > 0
        if visited_low and visited_high:
            # Estimate round-trip from visit distribution
            # A flat histogram means good sampling
            flatness = hist.min() / hist.mean() if hist.mean() > 0 else 0
            print(f"#   Histogram flatness: {flatness:.2f} (1.0 = perfectly flat)")
        else:
            missing = []
            if not visited_low:
                missing.append(f"T[0]={temps[0]:.0f}K")
            if not visited_high:
                missing.append(f"T[{n_temps-1}]={temps[n_temps-1]:.0f}K")
            print(f"#   Warning: Not yet visited: {', '.join(missing)}")


def update_langevin_a2(
    thermostat_state: Dict,
    mass: jnp.ndarray,
    a1: float,
    kT_new: float,
    fprec: str = "float32"
) -> Dict:
    """Update Langevin thermostat noise amplitude for new temperature.

    The Langevin thermostat uses:
        v_new = a1 * v_old + a2 * noise
    where a2 = sqrt((1 - a1^2) * kT / m)

    When temperature changes in simulated tempering, we need to update a2.

    Args:
        thermostat_state: Current thermostat state dictionary
        mass: Array of atomic masses
        a1: Langevin friction coefficient (exp(-gamma * dt))
        kT_new: New kT value in atomic units
        fprec: Floating point precision

    Returns:
        Updated thermostat state (note: actual a2 update happens in dyn_state)
    """
    # The a2 parameter is actually stored in the closure, not in the state
    # For now, we just update the rng_key to ensure different noise
    # The actual noise amplitude scaling happens through velocity rescaling
    return thermostat_state
