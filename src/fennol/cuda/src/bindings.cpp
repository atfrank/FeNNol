#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <sstream>
#include "../include/integrate.cuh"
#include "../include/restraints.cuh"
#include "../include/physics.cuh"
#include "../include/thermostats.cuh"
#include "../include/implicit_solvent.cuh"

namespace py = pybind11;

namespace fennol {
namespace cuda {

// RAII wrapper for CUDA device memory
template<typename T>
class CudaMemory {
    T* ptr = nullptr;
    size_t count = 0;

public:
    CudaMemory(size_t n) : count(n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        }
    }

    ~CudaMemory() {
        if (ptr) {
            cudaFree(ptr);  // Don't throw from destructor
        }
    }

    // Delete copy operations
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    // Allow move operations
    CudaMemory(CudaMemory&& other) noexcept : ptr(other.ptr), count(other.count) {
        other.ptr = nullptr;
        other.count = 0;
    }

    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    T* get() { return ptr; }
    const T* get() const { return ptr; }
    size_t size() const { return count; }

    void memset(int value) {
        if (ptr && count > 0) {
            CUDA_CHECK(cudaMemset(ptr, value, count * sizeof(T)));
        }
    }

    void copy_to_device(const void* host_ptr) {
        if (ptr && host_ptr && count > 0) {
            CUDA_CHECK(cudaMemcpy(ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void copy_from_device(void* host_ptr) const {
        if (ptr && host_ptr && count > 0) {
            CUDA_CHECK(cudaMemcpy(host_ptr, ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
};

// Input validation helpers
inline void validate_array_ndim(const py::buffer_info& buf, int expected_ndim, const std::string& name) {
    if (buf.ndim != expected_ndim) {
        std::ostringstream oss;
        oss << name << " must have " << expected_ndim << " dimensions, got " << buf.ndim;
        throw std::runtime_error(oss.str());
    }
}

inline void validate_array_shape_2d(const py::buffer_info& buf, ssize_t expected_dim1, const std::string& name) {
    validate_array_ndim(buf, 2, name);
    if (buf.shape[1] != expected_dim1) {
        std::ostringstream oss;
        oss << name << " must have shape (n, " << expected_dim1 << "), got (n, " << buf.shape[1] << ")";
        throw std::runtime_error(oss.str());
    }
}

inline void validate_array_size(const py::buffer_info& buf, ssize_t expected_size, const std::string& name) {
    if (buf.shape[0] != expected_size) {
        std::ostringstream oss;
        oss << name << " must have " << expected_size << " elements, got " << buf.shape[0];
        throw std::runtime_error(oss.str());
    }
}

inline void validate_positive(int value, const std::string& name) {
    if (value <= 0) {
        std::ostringstream oss;
        oss << name << " must be positive, got " << value;
        throw std::runtime_error(oss.str());
    }
}

// Wrapper functions for Python bindings

py::tuple py_velocity_verlet_step_a(
    py::array_t<double> coordinates,
    py::array_t<double> velocities,
    py::array_t<double> forces,
    py::array_t<double> masses,
    double dt
) {
    // Validate inputs
    auto coords_buf = coordinates.request();
    auto vel_buf = velocities.request();
    auto forces_buf = forces.request();
    auto masses_buf = masses.request();

    validate_array_shape_2d(coords_buf, 3, "coordinates");
    validate_array_shape_2d(vel_buf, 3, "velocities");
    validate_array_shape_2d(forces_buf, 3, "forces");
    validate_array_ndim(masses_buf, 1, "masses");

    int natoms = coords_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_array_size(vel_buf, natoms, "velocities");
    validate_array_size(forces_buf, natoms, "forces");
    validate_array_size(masses_buf, natoms, "masses");

    // Allocate device memory using RAII
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<double> d_vels(natoms * 3);
    CudaMemory<double> d_forces(natoms * 3);
    CudaMemory<double> d_masses(natoms);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_vels.copy_to_device(vel_buf.ptr);
    d_forces.copy_to_device(forces_buf.ptr);
    d_masses.copy_to_device(masses_buf.ptr);

    // Execute kernel
    velocity_verlet_step_a(d_coords.get(), d_vels.get(), d_forces.get(), d_masses.get(), dt, natoms);

    // Copy results back
    d_coords.copy_from_device(coords_buf.ptr);
    d_vels.copy_from_device(vel_buf.ptr);

    // RAII automatically frees memory
    return py::make_tuple(coordinates, velocities);
}

py::tuple py_velocity_verlet_step_b(
    py::array_t<double> velocities,
    py::array_t<double> forces,
    py::array_t<double> masses,
    double dt
) {
    // Validate inputs
    auto vel_buf = velocities.request();
    auto forces_buf = forces.request();
    auto masses_buf = masses.request();

    validate_array_shape_2d(vel_buf, 3, "velocities");
    validate_array_shape_2d(forces_buf, 3, "forces");
    validate_array_ndim(masses_buf, 1, "masses");

    int natoms = vel_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_array_size(forces_buf, natoms, "forces");
    validate_array_size(masses_buf, natoms, "masses");

    // Allocate device memory using RAII
    CudaMemory<double> d_vels(natoms * 3);
    CudaMemory<double> d_forces(natoms * 3);
    CudaMemory<double> d_masses(natoms);
    CudaMemory<double> d_ke(1);
    CudaMemory<double> d_ke_tensor(9);

    // Copy to device
    d_vels.copy_to_device(vel_buf.ptr);
    d_forces.copy_to_device(forces_buf.ptr);
    d_masses.copy_to_device(masses_buf.ptr);

    // Execute kernel
    velocity_verlet_step_b(d_vels.get(), d_forces.get(), d_masses.get(), dt, natoms,
                          d_ke.get(), d_ke_tensor.get());

    // Copy results back
    d_vels.copy_from_device(vel_buf.ptr);

    double kinetic_energy;
    auto ke_tensor = py::array_t<double>(9);
    auto ke_tensor_buf = ke_tensor.request();

    d_ke.copy_from_device(&kinetic_energy);
    d_ke_tensor.copy_from_device(ke_tensor_buf.ptr);

    // Reshape tensor to 3x3
    ke_tensor.resize({3, 3});

    // RAII automatically frees memory
    return py::make_tuple(velocities, kinetic_energy, ke_tensor);
}

py::tuple py_harmonic_distance_restraint(
    py::array_t<double> coordinates,
    py::array_t<int> atom_indices,
    py::array_t<double> target_distances,
    py::array_t<double> force_constants
) {
    // Validate inputs
    auto coords_buf = coordinates.request();
    auto indices_buf = atom_indices.request();
    auto targets_buf = target_distances.request();
    auto fcs_buf = force_constants.request();

    validate_array_shape_2d(coords_buf, 3, "coordinates");
    validate_array_shape_2d(indices_buf, 2, "atom_indices");
    validate_array_ndim(targets_buf, 1, "target_distances");
    validate_array_ndim(fcs_buf, 1, "force_constants");

    int natoms = coords_buf.shape[0];
    int nrestraints = indices_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_positive(nrestraints, "nrestraints");
    validate_array_size(targets_buf, nrestraints, "target_distances");
    validate_array_size(fcs_buf, nrestraints, "force_constants");

    // Allocate device memory using RAII
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<int> d_indices(nrestraints * 2);
    CudaMemory<double> d_targets(nrestraints);
    CudaMemory<double> d_fcs(nrestraints);
    CudaMemory<double> d_energy(1);
    CudaMemory<double> d_forces(natoms * 3);

    // Initialize forces to zero
    d_forces.memset(0);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_indices.copy_to_device(indices_buf.ptr);
    d_targets.copy_to_device(targets_buf.ptr);
    d_fcs.copy_to_device(fcs_buf.ptr);

    // Execute kernel
    harmonic_distance_restraint(d_coords.get(), d_indices.get(), d_targets.get(), d_fcs.get(),
                                natoms, nrestraints, d_energy.get(), d_forces.get());

    // Copy results back
    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    d_energy.copy_from_device(&energy);
    d_forces.copy_from_device(forces_buf.ptr);

    // RAII automatically frees memory
    return py::make_tuple(energy, forces);
}

// Similar wrappers for other restraint types...
py::tuple py_harmonic_angle_restraint(
    py::array_t<double> coordinates,
    py::array_t<int> atom_indices,
    py::array_t<double> target_angles,
    py::array_t<double> force_constants
) {
    // Validate inputs
    auto coords_buf = coordinates.request();
    auto indices_buf = atom_indices.request();
    auto targets_buf = target_angles.request();
    auto fcs_buf = force_constants.request();

    validate_array_shape_2d(coords_buf, 3, "coordinates");
    validate_array_shape_2d(indices_buf, 3, "atom_indices");
    validate_array_ndim(targets_buf, 1, "target_angles");
    validate_array_ndim(fcs_buf, 1, "force_constants");

    int natoms = coords_buf.shape[0];
    int nrestraints = indices_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_positive(nrestraints, "nrestraints");
    validate_array_size(targets_buf, nrestraints, "target_angles");
    validate_array_size(fcs_buf, nrestraints, "force_constants");

    // Allocate device memory using RAII
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<int> d_indices(nrestraints * 3);
    CudaMemory<double> d_targets(nrestraints);
    CudaMemory<double> d_fcs(nrestraints);
    CudaMemory<double> d_energy(1);
    CudaMemory<double> d_forces(natoms * 3);

    // Initialize forces to zero
    d_forces.memset(0);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_indices.copy_to_device(indices_buf.ptr);
    d_targets.copy_to_device(targets_buf.ptr);
    d_fcs.copy_to_device(fcs_buf.ptr);

    // Execute kernel
    harmonic_angle_restraint(d_coords.get(), d_indices.get(), d_targets.get(), d_fcs.get(),
                            natoms, nrestraints, d_energy.get(), d_forces.get());

    // Copy results back
    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    d_energy.copy_from_device(&energy);
    d_forces.copy_from_device(forces_buf.ptr);

    // RAII automatically frees memory
    return py::make_tuple(energy, forces);
}

py::tuple py_flat_bottom_distance_restraint(
    py::array_t<double> coordinates,
    py::array_t<int> atom_indices,
    py::array_t<double> target_distances,
    py::array_t<double> force_constants,
    py::array_t<double> tolerances
) {
    // Validate inputs
    auto coords_buf = coordinates.request();
    auto indices_buf = atom_indices.request();
    auto targets_buf = target_distances.request();
    auto fcs_buf = force_constants.request();
    auto tols_buf = tolerances.request();

    validate_array_shape_2d(coords_buf, 3, "coordinates");
    validate_array_shape_2d(indices_buf, 2, "atom_indices");
    validate_array_ndim(targets_buf, 1, "target_distances");
    validate_array_ndim(fcs_buf, 1, "force_constants");
    validate_array_ndim(tols_buf, 1, "tolerances");

    int natoms = coords_buf.shape[0];
    int nrestraints = indices_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_positive(nrestraints, "nrestraints");
    validate_array_size(targets_buf, nrestraints, "target_distances");
    validate_array_size(fcs_buf, nrestraints, "force_constants");
    validate_array_size(tols_buf, nrestraints, "tolerances");

    // Allocate device memory using RAII
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<int> d_indices(nrestraints * 2);
    CudaMemory<double> d_targets(nrestraints);
    CudaMemory<double> d_fcs(nrestraints);
    CudaMemory<double> d_tols(nrestraints);
    CudaMemory<double> d_energy(1);
    CudaMemory<double> d_forces(natoms * 3);

    // Initialize forces to zero
    d_forces.memset(0);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_indices.copy_to_device(indices_buf.ptr);
    d_targets.copy_to_device(targets_buf.ptr);
    d_fcs.copy_to_device(fcs_buf.ptr);
    d_tols.copy_to_device(tols_buf.ptr);

    // Execute kernel
    flat_bottom_distance_restraint(d_coords.get(), d_indices.get(), d_targets.get(), d_fcs.get(), d_tols.get(),
                                    natoms, nrestraints, d_energy.get(), d_forces.get());

    // Copy results back
    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    d_energy.copy_from_device(&energy);
    d_forces.copy_from_device(forces_buf.ptr);

    // RAII automatically frees memory
    return py::make_tuple(energy, forces);
}

py::tuple py_nlh_repulsion(
    py::array_t<double> coordinates,
    py::array_t<int> atomic_numbers,
    py::array_t<int> atom_pairs,
    py::array_t<double> pair_coefficients,
    double cutoff
) {
    // Validate inputs
    auto coords_buf = coordinates.request();
    auto Z_buf = atomic_numbers.request();
    auto pairs_buf = atom_pairs.request();
    auto coeffs_buf = pair_coefficients.request();

    validate_array_shape_2d(coords_buf, 3, "coordinates");
    validate_array_ndim(Z_buf, 1, "atomic_numbers");
    validate_array_shape_2d(pairs_buf, 2, "atom_pairs");
    validate_array_shape_2d(coeffs_buf, 6, "pair_coefficients");

    int natoms = coords_buf.shape[0];
    int npairs = pairs_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_positive(npairs, "npairs");
    validate_array_size(Z_buf, natoms, "atomic_numbers");
    validate_array_size(coeffs_buf, npairs, "pair_coefficients");

    // Allocate device memory using RAII
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<int> d_Z(natoms);
    CudaMemory<int> d_pairs(npairs * 2);
    CudaMemory<double> d_coeffs(npairs * 6);
    CudaMemory<double> d_energy(1);
    CudaMemory<double> d_forces(natoms * 3);

    // Initialize forces to zero
    d_forces.memset(0);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_Z.copy_to_device(Z_buf.ptr);
    d_pairs.copy_to_device(pairs_buf.ptr);
    d_coeffs.copy_to_device(coeffs_buf.ptr);

    // Execute kernel
    nlh_repulsion(d_coords.get(), d_Z.get(), d_pairs.get(), d_coeffs.get(), natoms, npairs, cutoff,
                  d_energy.get(), d_forces.get());

    // Copy results back
    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    d_energy.copy_from_device(&energy);
    d_forces.copy_from_device(forces_buf.ptr);

    // RAII automatically frees memory
    return py::make_tuple(energy, forces);
}

py::tuple py_berendsen_thermostat(
    py::array_t<double> velocities,
    py::array_t<double> masses,
    double target_temperature,
    double coupling_time,
    double dt
) {
    // Validate inputs
    auto vel_buf = velocities.request();
    auto masses_buf = masses.request();

    validate_array_shape_2d(vel_buf, 3, "velocities");
    validate_array_ndim(masses_buf, 1, "masses");

    int natoms = vel_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_array_size(masses_buf, natoms, "masses");

    // Allocate device memory using RAII
    CudaMemory<double> d_vels(natoms * 3);
    CudaMemory<double> d_masses(natoms);

    // Copy to device
    d_vels.copy_to_device(vel_buf.ptr);
    d_masses.copy_to_device(masses_buf.ptr);

    // Execute kernel
    double current_temp;
    berendsen_thermostat(d_vels.get(), d_masses.get(), natoms, target_temperature, coupling_time, dt, &current_temp);

    // Copy results back
    d_vels.copy_from_device(vel_buf.ptr);

    // RAII automatically frees memory
    return py::make_tuple(velocities, current_temp);
}

py::tuple py_velocity_rescale_thermostat(
    py::array_t<double> velocities,
    py::array_t<double> masses,
    double target_temperature
) {
    // Validate inputs
    auto vel_buf = velocities.request();
    auto masses_buf = masses.request();

    validate_array_shape_2d(vel_buf, 3, "velocities");
    validate_array_ndim(masses_buf, 1, "masses");

    int natoms = vel_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_array_size(masses_buf, natoms, "masses");

    // Allocate device memory using RAII
    CudaMemory<double> d_vels(natoms * 3);
    CudaMemory<double> d_masses(natoms);

    // Copy to device
    d_vels.copy_to_device(vel_buf.ptr);
    d_masses.copy_to_device(masses_buf.ptr);

    // Execute kernel
    double current_temp;
    velocity_rescale_thermostat(d_vels.get(), d_masses.get(), natoms, target_temperature, &current_temp);

    // Copy results back
    d_vels.copy_from_device(vel_buf.ptr);

    // RAII automatically frees memory
    return py::make_tuple(velocities, current_temp);
}

py::tuple py_backside_attack_restraint(
    py::array_t<double> coordinates,
    py::array_t<int> restraint_indices,
    py::array_t<double> target_angles,
    py::array_t<double> angle_force_constants,
    py::array_t<double> target_distances,
    py::array_t<double> distance_force_constants
) {
    // Validate inputs
    auto coords_buf = coordinates.request();
    auto indices_buf = restraint_indices.request();
    auto angles_buf = target_angles.request();
    auto angle_fcs_buf = angle_force_constants.request();
    auto dists_buf = target_distances.request();
    auto dist_fcs_buf = distance_force_constants.request();

    validate_array_shape_2d(coords_buf, 3, "coordinates");
    validate_array_shape_2d(indices_buf, 3, "restraint_indices");
    validate_array_ndim(angles_buf, 1, "target_angles");
    validate_array_ndim(angle_fcs_buf, 1, "angle_force_constants");
    validate_array_ndim(dists_buf, 1, "target_distances");
    validate_array_ndim(dist_fcs_buf, 1, "distance_force_constants");

    int natoms = coords_buf.shape[0];
    int nrestraints = indices_buf.shape[0];
    validate_positive(natoms, "natoms");
    validate_positive(nrestraints, "nrestraints");
    validate_array_size(angles_buf, nrestraints, "target_angles");
    validate_array_size(angle_fcs_buf, nrestraints, "angle_force_constants");
    validate_array_size(dists_buf, nrestraints, "target_distances");
    validate_array_size(dist_fcs_buf, nrestraints, "distance_force_constants");

    // Allocate device memory using RAII
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<int> d_indices(nrestraints * 3);
    CudaMemory<double> d_angles(nrestraints);
    CudaMemory<double> d_angle_fcs(nrestraints);
    CudaMemory<double> d_dists(nrestraints);
    CudaMemory<double> d_dist_fcs(nrestraints);
    CudaMemory<double> d_energy(1);
    CudaMemory<double> d_forces(natoms * 3);

    // Initialize forces to zero
    d_forces.memset(0);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_indices.copy_to_device(indices_buf.ptr);
    d_angles.copy_to_device(angles_buf.ptr);
    d_angle_fcs.copy_to_device(angle_fcs_buf.ptr);
    d_dists.copy_to_device(dists_buf.ptr);
    d_dist_fcs.copy_to_device(dist_fcs_buf.ptr);

    // Execute kernel
    backside_attack_restraint(d_coords.get(), d_indices.get(), d_angles.get(), d_angle_fcs.get(),
                              d_dists.get(), d_dist_fcs.get(), natoms, nrestraints,
                              d_energy.get(), d_forces.get());

    // Copy results back
    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    d_energy.copy_from_device(&energy);
    d_forces.copy_from_device(forces_buf.ptr);

    // RAII automatically frees memory
    return py::make_tuple(energy, forces);
}

// ===== Implicit Solvent Python Wrappers =====

py::array_t<double> py_gb_compute_born_radii(
    py::array_t<double> coordinates,
    py::array_t<double> intrinsic_radii,
    py::array_t<double> b_params,
    py::array_t<double> c_params,
    double cutoff
) {
    auto coords_buf = coordinates.request();
    auto radii_buf = intrinsic_radii.request();
    auto b_buf = b_params.request();
    auto c_buf = c_params.request();

    // Validate inputs
    validate_array_shape_2d(coords_buf, 3, "coordinates");
    int natoms = coords_buf.shape[0];

    validate_array_size(radii_buf, natoms, "intrinsic_radii");
    validate_array_size(b_buf, natoms, "b_params");
    validate_array_size(c_buf, natoms, "c_params");

    // Allocate device memory
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<double> d_radii(natoms);
    CudaMemory<double> d_b_params(natoms);
    CudaMemory<double> d_c_params(natoms);
    CudaMemory<double> d_born_radii(natoms);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_radii.copy_to_device(radii_buf.ptr);
    d_b_params.copy_to_device(b_buf.ptr);
    d_c_params.copy_to_device(c_buf.ptr);

    // Call CUDA kernel
    fennol::cuda::implicit_solvent::compute_born_radii_obc(
        natoms,
        d_coords.get(),
        d_radii.get(),
        d_b_params.get(),
        d_c_params.get(),
        cutoff,
        d_born_radii.get()
    );

    // Create output array
    auto born_radii = py::array_t<double>(natoms);
    auto born_radii_buf = born_radii.request();

    // Copy from device
    d_born_radii.copy_from_device(born_radii_buf.ptr);

    return born_radii;
}

py::tuple py_gb_compute_energy_forces(
    py::array_t<double> coordinates,
    py::array_t<double> charges,
    py::array_t<double> born_radii,
    double dielectric,
    double cutoff
) {
    auto coords_buf = coordinates.request();
    auto charges_buf = charges.request();
    auto born_radii_buf = born_radii.request();

    // Validate inputs
    validate_array_shape_2d(coords_buf, 3, "coordinates");
    int natoms = coords_buf.shape[0];

    validate_array_size(charges_buf, natoms, "charges");
    validate_array_size(born_radii_buf, natoms, "born_radii");

    // Allocate device memory
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<double> d_charges(natoms);
    CudaMemory<double> d_born_radii(natoms);
    CudaMemory<double> d_energy(1);
    CudaMemory<double> d_forces(natoms * 3);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_charges.copy_to_device(charges_buf.ptr);
    d_born_radii.copy_to_device(born_radii_buf.ptr);

    // Call CUDA kernel
    fennol::cuda::implicit_solvent::compute_gb_energy_forces(
        natoms,
        d_coords.get(),
        d_charges.get(),
        d_born_radii.get(),
        dielectric,
        cutoff,
        d_energy.get(),
        d_forces.get()
    );

    // Create output arrays
    auto energy = py::array_t<double>(1);
    auto forces = py::array_t<double>({natoms, 3});

    auto energy_buf = energy.request();
    auto forces_buf = forces.request();

    // Copy from device
    d_energy.copy_from_device(energy_buf.ptr);
    d_forces.copy_from_device(forces_buf.ptr);

    return py::make_tuple(energy, forces);
}

py::tuple py_gb_compute_nonpolar(
    py::array_t<double> coordinates,
    py::array_t<double> born_radii,
    py::array_t<double> gamma_params,
    double probe_radius
) {
    auto coords_buf = coordinates.request();
    auto born_radii_buf = born_radii.request();
    auto gamma_buf = gamma_params.request();

    // Validate inputs
    validate_array_shape_2d(coords_buf, 3, "coordinates");
    int natoms = coords_buf.shape[0];

    validate_array_size(born_radii_buf, natoms, "born_radii");
    validate_array_size(gamma_buf, natoms, "gamma_params");

    // Allocate device memory
    CudaMemory<double> d_coords(natoms * 3);
    CudaMemory<double> d_born_radii(natoms);
    CudaMemory<double> d_gamma(natoms);
    CudaMemory<double> d_energy(1);
    CudaMemory<double> d_forces(natoms * 3);

    // Copy to device
    d_coords.copy_to_device(coords_buf.ptr);
    d_born_radii.copy_to_device(born_radii_buf.ptr);
    d_gamma.copy_to_device(gamma_buf.ptr);

    // Initialize forces to zero
    d_forces.memset(0);

    // Call CUDA kernel
    fennol::cuda::implicit_solvent::compute_nonpolar_sasa(
        natoms,
        d_coords.get(),
        d_born_radii.get(),
        d_gamma.get(),
        probe_radius,
        d_energy.get(),
        d_forces.get()
    );

    // Create output arrays
    auto energy = py::array_t<double>(1);
    auto forces = py::array_t<double>({natoms, 3});

    auto energy_buf = energy.request();
    auto forces_buf = forces.request();

    // Copy from device
    d_energy.copy_from_device(energy_buf.ptr);
    d_forces.copy_from_device(forces_buf.ptr);

    return py::make_tuple(energy, forces);
}

} // namespace cuda
} // namespace fennol

PYBIND11_MODULE(fennol_cuda, m) {
    m.doc() = "CUDA-accelerated molecular dynamics kernels for FeNNol";

    // Integration functions
    m.def("velocity_verlet_step_a", &fennol::cuda::py_velocity_verlet_step_a,
          "Velocity Verlet integration step A",
          py::arg("coordinates"), py::arg("velocities"), py::arg("forces"),
          py::arg("masses"), py::arg("dt"));

    m.def("velocity_verlet_step_b", &fennol::cuda::py_velocity_verlet_step_b,
          "Velocity Verlet integration step B",
          py::arg("velocities"), py::arg("forces"), py::arg("masses"), py::arg("dt"));

    // Restraint functions
    m.def("harmonic_distance_restraint", &fennol::cuda::py_harmonic_distance_restraint,
          "Harmonic distance restraint",
          py::arg("coordinates"), py::arg("atom_indices"),
          py::arg("target_distances"), py::arg("force_constants"));

    m.def("harmonic_angle_restraint", &fennol::cuda::py_harmonic_angle_restraint,
          "Harmonic angle restraint",
          py::arg("coordinates"), py::arg("atom_indices"),
          py::arg("target_angles"), py::arg("force_constants"));

    m.def("flat_bottom_distance_restraint", &fennol::cuda::py_flat_bottom_distance_restraint,
          "Flat-bottom distance restraint",
          py::arg("coordinates"), py::arg("atom_indices"),
          py::arg("target_distances"), py::arg("force_constants"), py::arg("tolerances"));

    // Physics functions
    m.def("nlh_repulsion", &fennol::cuda::py_nlh_repulsion,
          "NLH (Nordlund-Lehtola-Hobler) repulsion potential",
          py::arg("coordinates"), py::arg("atomic_numbers"), py::arg("atom_pairs"),
          py::arg("pair_coefficients"), py::arg("cutoff"));

    // Thermostat functions
    m.def("berendsen_thermostat", &fennol::cuda::py_berendsen_thermostat,
          "Berendsen weak coupling thermostat",
          py::arg("velocities"), py::arg("masses"), py::arg("target_temperature"),
          py::arg("coupling_time"), py::arg("dt"));

    m.def("velocity_rescale_thermostat", &fennol::cuda::py_velocity_rescale_thermostat,
          "Velocity rescaling thermostat",
          py::arg("velocities"), py::arg("masses"), py::arg("target_temperature"));

    m.def("backside_attack_restraint", &fennol::cuda::py_backside_attack_restraint,
          "Backside attack restraint for SN2 reactions (combines angle and distance restraints)",
          py::arg("coordinates"), py::arg("restraint_indices"), py::arg("target_angles"),
          py::arg("angle_force_constants"), py::arg("target_distances"), py::arg("distance_force_constants"));

    // Implicit solvent functions
    m.def("gb_compute_born_radii", &fennol::cuda::py_gb_compute_born_radii,
          "Compute Born radii using OBC model",
          py::arg("coordinates"), py::arg("intrinsic_radii"), py::arg("b_params"),
          py::arg("c_params"), py::arg("cutoff"));

    m.def("gb_compute_energy_forces", &fennol::cuda::py_gb_compute_energy_forces,
          "Compute GB electrostatic energy and forces",
          py::arg("coordinates"), py::arg("charges"), py::arg("born_radii"),
          py::arg("dielectric"), py::arg("cutoff"));

    m.def("gb_compute_nonpolar", &fennol::cuda::py_gb_compute_nonpolar,
          "Compute non-polar (surface area) energy and forces",
          py::arg("coordinates"), py::arg("born_radii"), py::arg("gamma_params"),
          py::arg("probe_radius"));
}
