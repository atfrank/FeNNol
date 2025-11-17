#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/integrate.cuh"
#include "../include/restraints.cuh"
#include "../include/physics.cuh"
#include "../include/thermostats.cuh"

namespace py = pybind11;

namespace fennol {
namespace cuda {

// Helper function to get device pointer from numpy array
template<typename T>
T* get_device_ptr(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    return static_cast<T*>(buf.ptr);
}

// Wrapper functions for Python bindings

py::tuple py_velocity_verlet_step_a(
    py::array_t<double> coordinates,
    py::array_t<double> velocities,
    py::array_t<double> forces,
    py::array_t<double> masses,
    double dt
) {
    auto coords_buf = coordinates.request();
    auto vel_buf = velocities.request();
    auto forces_buf = forces.request();
    auto masses_buf = masses.request();

    if (coords_buf.ndim != 2 || coords_buf.shape[1] != 3) {
        throw std::runtime_error("coordinates must be (natoms, 3)");
    }
    if (vel_buf.ndim != 2 || vel_buf.shape[1] != 3) {
        throw std::runtime_error("velocities must be (natoms, 3)");
    }

    int natoms = coords_buf.shape[0];

    // Allocate device memory and copy data
    double *d_coords, *d_vels, *d_forces, *d_masses;
    CUDA_CHECK(cudaMalloc(&d_coords, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vels, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forces, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_masses, natoms * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_coords, coords_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vels, vel_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forces, forces_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_masses, masses_buf.ptr, natoms * sizeof(double), cudaMemcpyHostToDevice));

    // Execute kernel
    velocity_verlet_step_a(d_coords, d_vels, d_forces, d_masses, dt, natoms);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(coords_buf.ptr, d_coords, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel_buf.ptr, d_vels, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_vels));
    CUDA_CHECK(cudaFree(d_forces));
    CUDA_CHECK(cudaFree(d_masses));

    return py::make_tuple(coordinates, velocities);
}

py::tuple py_velocity_verlet_step_b(
    py::array_t<double> velocities,
    py::array_t<double> forces,
    py::array_t<double> masses,
    double dt
) {
    auto vel_buf = velocities.request();
    auto forces_buf = forces.request();
    auto masses_buf = masses.request();

    int natoms = vel_buf.shape[0];

    // Allocate device memory
    double *d_vels, *d_forces, *d_masses, *d_ke, *d_ke_tensor;
    CUDA_CHECK(cudaMalloc(&d_vels, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forces, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_masses, natoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ke, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ke_tensor, 9 * sizeof(double)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_vels, vel_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forces, forces_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_masses, masses_buf.ptr, natoms * sizeof(double), cudaMemcpyHostToDevice));

    // Execute kernel
    velocity_verlet_step_b(d_vels, d_forces, d_masses, dt, natoms, d_ke, d_ke_tensor);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(vel_buf.ptr, d_vels, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    double kinetic_energy;
    auto ke_tensor = py::array_t<double>(9);
    auto ke_tensor_buf = ke_tensor.request();

    CUDA_CHECK(cudaMemcpy(&kinetic_energy, d_ke, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ke_tensor_buf.ptr, d_ke_tensor, 9 * sizeof(double), cudaMemcpyDeviceToHost));

    // Reshape tensor to 3x3
    ke_tensor.resize({3, 3});

    // Free device memory
    CUDA_CHECK(cudaFree(d_vels));
    CUDA_CHECK(cudaFree(d_forces));
    CUDA_CHECK(cudaFree(d_masses));
    CUDA_CHECK(cudaFree(d_ke));
    CUDA_CHECK(cudaFree(d_ke_tensor));

    return py::make_tuple(velocities, kinetic_energy, ke_tensor);
}

py::tuple py_harmonic_distance_restraint(
    py::array_t<double> coordinates,
    py::array_t<int> atom_indices,
    py::array_t<double> target_distances,
    py::array_t<double> force_constants
) {
    auto coords_buf = coordinates.request();
    auto indices_buf = atom_indices.request();
    auto targets_buf = target_distances.request();
    auto fcs_buf = force_constants.request();

    int natoms = coords_buf.shape[0];
    int nrestraints = indices_buf.shape[0];

    // Allocate device memory
    double *d_coords, *d_targets, *d_fcs, *d_energy, *d_forces;
    int *d_indices;

    CUDA_CHECK(cudaMalloc(&d_coords, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_indices, nrestraints * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, nrestraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fcs, nrestraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forces, natoms * 3 * sizeof(double)));

    // Initialize forces to zero
    CUDA_CHECK(cudaMemset(d_forces, 0, natoms * 3 * sizeof(double)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_coords, coords_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, indices_buf.ptr, nrestraints * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets_buf.ptr, nrestraints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fcs, fcs_buf.ptr, nrestraints * sizeof(double), cudaMemcpyHostToDevice));

    // Execute kernel
    harmonic_distance_restraint(d_coords, d_indices, d_targets, d_fcs,
                                natoms, nrestraints, d_energy, d_forces);

    // Copy results back
    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    CUDA_CHECK(cudaMemcpy(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forces_buf.ptr, d_forces, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_fcs));
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_forces));

    return py::make_tuple(energy, forces);
}

// Similar wrappers for other restraint types...
py::tuple py_harmonic_angle_restraint(
    py::array_t<double> coordinates,
    py::array_t<int> atom_indices,
    py::array_t<double> target_angles,
    py::array_t<double> force_constants
) {
    auto coords_buf = coordinates.request();
    auto indices_buf = atom_indices.request();
    auto targets_buf = target_angles.request();
    auto fcs_buf = force_constants.request();

    int natoms = coords_buf.shape[0];
    int nrestraints = indices_buf.shape[0];

    double *d_coords, *d_targets, *d_fcs, *d_energy, *d_forces;
    int *d_indices;

    CUDA_CHECK(cudaMalloc(&d_coords, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_indices, nrestraints * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, nrestraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fcs, nrestraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forces, natoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_forces, 0, natoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_coords, coords_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, indices_buf.ptr, nrestraints * 3 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets_buf.ptr, nrestraints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fcs, fcs_buf.ptr, nrestraints * sizeof(double), cudaMemcpyHostToDevice));

    harmonic_angle_restraint(d_coords, d_indices, d_targets, d_fcs,
                            natoms, nrestraints, d_energy, d_forces);

    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    CUDA_CHECK(cudaMemcpy(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forces_buf.ptr, d_forces, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_fcs));
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_forces));

    return py::make_tuple(energy, forces);
}

py::tuple py_flat_bottom_distance_restraint(
    py::array_t<double> coordinates,
    py::array_t<int> atom_indices,
    py::array_t<double> target_distances,
    py::array_t<double> force_constants,
    py::array_t<double> tolerances
) {
    auto coords_buf = coordinates.request();
    auto indices_buf = atom_indices.request();
    auto targets_buf = target_distances.request();
    auto fcs_buf = force_constants.request();
    auto tols_buf = tolerances.request();

    int natoms = coords_buf.shape[0];
    int nrestraints = indices_buf.shape[0];

    double *d_coords, *d_targets, *d_fcs, *d_tols, *d_energy, *d_forces;
    int *d_indices;

    CUDA_CHECK(cudaMalloc(&d_coords, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_indices, nrestraints * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, nrestraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fcs, nrestraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tols, nrestraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forces, natoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_forces, 0, natoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_coords, coords_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, indices_buf.ptr, nrestraints * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets_buf.ptr, nrestraints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fcs, fcs_buf.ptr, nrestraints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tols, tols_buf.ptr, nrestraints * sizeof(double), cudaMemcpyHostToDevice));

    flat_bottom_distance_restraint(d_coords, d_indices, d_targets, d_fcs, d_tols,
                                    natoms, nrestraints, d_energy, d_forces);

    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    CUDA_CHECK(cudaMemcpy(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forces_buf.ptr, d_forces, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_fcs));
    CUDA_CHECK(cudaFree(d_tols));
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_forces));

    return py::make_tuple(energy, forces);
}

py::tuple py_nlh_repulsion(
    py::array_t<double> coordinates,
    py::array_t<int> atomic_numbers,
    py::array_t<int> atom_pairs,
    py::array_t<double> pair_coefficients,
    double cutoff
) {
    auto coords_buf = coordinates.request();
    auto Z_buf = atomic_numbers.request();
    auto pairs_buf = atom_pairs.request();
    auto coeffs_buf = pair_coefficients.request();

    int natoms = coords_buf.shape[0];
    int npairs = pairs_buf.shape[0];

    double *d_coords, *d_coeffs, *d_energy, *d_forces;
    int *d_Z, *d_pairs;

    CUDA_CHECK(cudaMalloc(&d_coords, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Z, natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pairs, npairs * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coeffs, npairs * 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forces, natoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_forces, 0, natoms * 3 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_coords, coords_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Z, Z_buf.ptr, natoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pairs, pairs_buf.ptr, npairs * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeffs, coeffs_buf.ptr, npairs * 6 * sizeof(double), cudaMemcpyHostToDevice));

    nlh_repulsion(d_coords, d_Z, d_pairs, d_coeffs, natoms, npairs, cutoff, d_energy, d_forces);

    double energy;
    auto forces = py::array_t<double>({natoms, 3});
    auto forces_buf = forces.request();

    CUDA_CHECK(cudaMemcpy(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forces_buf.ptr, d_forces, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_Z));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_coeffs));
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_forces));

    return py::make_tuple(energy, forces);
}

py::tuple py_berendsen_thermostat(
    py::array_t<double> velocities,
    py::array_t<double> masses,
    double target_temperature,
    double coupling_time,
    double dt
) {
    auto vel_buf = velocities.request();
    auto masses_buf = masses.request();

    int natoms = vel_buf.shape[0];

    double *d_vels, *d_masses;
    CUDA_CHECK(cudaMalloc(&d_vels, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_masses, natoms * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_vels, vel_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_masses, masses_buf.ptr, natoms * sizeof(double), cudaMemcpyHostToDevice));

    double current_temp;
    berendsen_thermostat(d_vels, d_masses, natoms, target_temperature, coupling_time, dt, &current_temp);

    CUDA_CHECK(cudaMemcpy(vel_buf.ptr, d_vels, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_vels));
    CUDA_CHECK(cudaFree(d_masses));

    return py::make_tuple(velocities, current_temp);
}

py::tuple py_velocity_rescale_thermostat(
    py::array_t<double> velocities,
    py::array_t<double> masses,
    double target_temperature
) {
    auto vel_buf = velocities.request();
    auto masses_buf = masses.request();

    int natoms = vel_buf.shape[0];

    double *d_vels, *d_masses;
    CUDA_CHECK(cudaMalloc(&d_vels, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_masses, natoms * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_vels, vel_buf.ptr, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_masses, masses_buf.ptr, natoms * sizeof(double), cudaMemcpyHostToDevice));

    double current_temp;
    velocity_rescale_thermostat(d_vels, d_masses, natoms, target_temperature, &current_temp);

    CUDA_CHECK(cudaMemcpy(vel_buf.ptr, d_vels, natoms * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_vels));
    CUDA_CHECK(cudaFree(d_masses));

    return py::make_tuple(velocities, current_temp);
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
}
