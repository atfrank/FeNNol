#include "../include/restraints.cuh"
#include <cmath>

namespace fennol {
namespace cuda {

// Harmonic distance restraint kernel
__global__ void harmonic_distance_restraint_kernel(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int nrestraints,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrestraints) return;

    int i = atom_indices[idx * 2 + 0];
    int j = atom_indices[idx * 2 + 1];

    // Get positions
    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    // Compute distance
    Vec3 rij = rj - ri;
    double r = rij.norm();

    // Compute energy and force
    double r0 = target_distances[idx];
    double k = force_constants[idx];
    double dr = r - r0;

    double energy = 0.5 * k * dr * dr;
    partial_energies[idx] = energy;

    // Force: F = -k * (r - r0) * (rij / r)
    // Force on atom i: F_i = k * dr * (rij / r)
    // Force on atom j: F_j = -F_i
    if (r > 1e-10) {
        Vec3 force_dir = rij * (k * dr / r);

        // Atomic add to force arrays
        for (int d = 0; d < 3; ++d) {
            double f = (d == 0) ? force_dir.x : (d == 1) ? force_dir.y : force_dir.z;
            atomicAddDouble(&forces[i * 3 + d], f);
            atomicAddDouble(&forces[j * 3 + d], -f);
        }
    }
}

void harmonic_distance_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
) {
    // Allocate temporary storage for partial energies
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nrestraints * sizeof(double)));

    int nblocks = (nrestraints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    harmonic_distance_restraint_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_indices, target_distances, force_constants,
        nrestraints, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    // Reduce energies
    double* h_partial_energies = new double[nrestraints];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nrestraints * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nrestraints; ++i) {
        total_energy += h_partial_energies[i];
    }
    CUDA_CHECK(cudaMemcpy(energy, &total_energy, sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Lower distance restraint kernel
__global__ void lower_distance_restraint_kernel(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int nrestraints,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrestraints) return;

    int i = atom_indices[idx * 2 + 0];
    int j = atom_indices[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    double r0 = target_distances[idx];
    double k = force_constants[idx];
    double violation = fmax(0.0, r0 - r);

    double energy = 0.5 * k * violation * violation;
    partial_energies[idx] = energy;

    if (violation > 1e-10 && r > 1e-10) {
        // Force: F = -k * (r0 - r) * (rij / r)
        Vec3 force_dir = rij * (-k * violation / r);

        for (int d = 0; d < 3; ++d) {
            double f = (d == 0) ? force_dir.x : (d == 1) ? force_dir.y : force_dir.z;
            atomicAddDouble(&forces[i * 3 + d], f);
            atomicAddDouble(&forces[j * 3 + d], -f);
        }
    }
}

void lower_distance_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nrestraints * sizeof(double)));

    int nblocks = (nrestraints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    lower_distance_restraint_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_indices, target_distances, force_constants,
        nrestraints, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[nrestraints];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nrestraints * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nrestraints; ++i) {
        total_energy += h_partial_energies[i];
    }
    CUDA_CHECK(cudaMemcpy(energy, &total_energy, sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Upper distance restraint kernel
__global__ void upper_distance_restraint_kernel(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int nrestraints,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrestraints) return;

    int i = atom_indices[idx * 2 + 0];
    int j = atom_indices[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    double r0 = target_distances[idx];
    double k = force_constants[idx];
    double violation = fmax(0.0, r - r0);

    double energy = 0.5 * k * violation * violation;
    partial_energies[idx] = energy;

    if (violation > 1e-10 && r > 1e-10) {
        Vec3 force_dir = rij * (k * violation / r);

        for (int d = 0; d < 3; ++d) {
            double f = (d == 0) ? force_dir.x : (d == 1) ? force_dir.y : force_dir.z;
            atomicAddDouble(&forces[i * 3 + d], f);
            atomicAddDouble(&forces[j * 3 + d], -f);
        }
    }
}

void upper_distance_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nrestraints * sizeof(double)));

    int nblocks = (nrestraints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    upper_distance_restraint_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_indices, target_distances, force_constants,
        nrestraints, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[nrestraints];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nrestraints * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nrestraints; ++i) {
        total_energy += h_partial_energies[i];
    }
    CUDA_CHECK(cudaMemcpy(energy, &total_energy, sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Flat-bottom distance restraint kernel
__global__ void flat_bottom_distance_restraint_kernel(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    const double* tolerances,
    int nrestraints,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrestraints) return;

    int i = atom_indices[idx * 2 + 0];
    int j = atom_indices[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    double r0 = target_distances[idx];
    double k = force_constants[idx];
    double tol = tolerances[idx];

    // Calculate deviation from target
    double diff = r - r0;
    double abs_diff = fabs(diff);

    // Calculate violation (how far outside the flat region)
    double violation = fmax(0.0, abs_diff - tol);

    // Energy is 0.5*k*violation^2
    double energy = 0.5 * k * violation * violation;
    partial_energies[idx] = energy;

    // Force is -k*violation*sign(diff) (negative of energy gradient)
    // Only non-zero when outside the tolerance
    if (violation > 1e-10 && r > 1e-10) {
        double sign = (diff > 0) ? -1.0 : 1.0;  // Inverted to pull atoms together when too far
        double force_mag = k * violation * sign;
        Vec3 force_dir = rij * (force_mag / r);

        for (int d = 0; d < 3; ++d) {
            double f = (d == 0) ? force_dir.x : (d == 1) ? force_dir.y : force_dir.z;
            atomicAddDouble(&forces[i * 3 + d], f);
            atomicAddDouble(&forces[j * 3 + d], -f);
        }
    }
}

void flat_bottom_distance_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    const double* tolerances,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nrestraints * sizeof(double)));

    int nblocks = (nrestraints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    flat_bottom_distance_restraint_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_indices, target_distances, force_constants, tolerances,
        nrestraints, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[nrestraints];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nrestraints * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nrestraints; ++i) {
        total_energy += h_partial_energies[i];
    }
    CUDA_CHECK(cudaMemcpy(energy, &total_energy, sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Harmonic angle restraint kernel
__global__ void harmonic_angle_restraint_kernel(
    const double* coordinates,
    const int* atom_indices,
    const double* target_angles,
    const double* force_constants,
    int nrestraints,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrestraints) return;

    int i = atom_indices[idx * 3 + 0];
    int j = atom_indices[idx * 3 + 1];  // central atom
    int k = atom_indices[idx * 3 + 2];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);
    Vec3 rk(coordinates[k * 3 + 0], coordinates[k * 3 + 1], coordinates[k * 3 + 2]);

    Vec3 rji = ri - rj;
    Vec3 rjk = rk - rj;

    double r_ji = rji.norm();
    double r_jk = rjk.norm();

    if (r_ji < 1e-10 || r_jk < 1e-10) {
        partial_energies[idx] = 0.0;
        return;
    }

    // Compute angle
    double cos_theta = rji.dot(rjk) / (r_ji * r_jk);
    cos_theta = fmax(-1.0, fmin(1.0, cos_theta));  // clamp
    double theta = acos(cos_theta);

    double theta0 = target_angles[idx];
    double force_constant = force_constants[idx];
    double dtheta = theta - theta0;

    double energy = 0.5 * force_constant * dtheta * dtheta;
    partial_energies[idx] = energy;

    // Compute forces
    double sin_theta = sin(theta);
    if (fabs(sin_theta) < 1e-10) {
        return;  // No force at 0 or 180 degrees
    }

    double prefactor = -force_constant * dtheta / sin_theta;

    Vec3 di = (rjk * (1.0 / (r_ji * r_jk)) - rji * (cos_theta / (r_ji * r_ji))) * prefactor;
    Vec3 dk = (rji * (1.0 / (r_ji * r_jk)) - rjk * (cos_theta / (r_jk * r_jk))) * prefactor;
    Vec3 dj = (di + dk) * (-1.0);

    // Atomic add to force arrays
    for (int d = 0; d < 3; ++d) {
        double fi = (d == 0) ? di.x : (d == 1) ? di.y : di.z;
        double fj = (d == 0) ? dj.x : (d == 1) ? dj.y : dj.z;
        double fk = (d == 0) ? dk.x : (d == 1) ? dk.y : dk.z;

        atomicAddDouble(&forces[i * 3 + d], fi);
        atomicAddDouble(&forces[j * 3 + d], fj);
        atomicAddDouble(&forces[k * 3 + d], fk);
    }
}

void harmonic_angle_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_angles,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nrestraints * sizeof(double)));

    int nblocks = (nrestraints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    harmonic_angle_restraint_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_indices, target_angles, force_constants,
        nrestraints, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[nrestraints];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nrestraints * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nrestraints; ++i) {
        total_energy += h_partial_energies[i];
    }
    CUDA_CHECK(cudaMemcpy(energy, &total_energy, sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Harmonic dihedral restraint kernel
__global__ void harmonic_dihedral_restraint_kernel(
    const double* coordinates,
    const int* atom_indices,
    const double* target_dihedrals,
    const double* force_constants,
    int nrestraints,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrestraints) return;

    int i = atom_indices[idx * 4 + 0];
    int j = atom_indices[idx * 4 + 1];
    int k = atom_indices[idx * 4 + 2];
    int l = atom_indices[idx * 4 + 3];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);
    Vec3 rk(coordinates[k * 3 + 0], coordinates[k * 3 + 1], coordinates[k * 3 + 2]);
    Vec3 rl(coordinates[l * 3 + 0], coordinates[l * 3 + 1], coordinates[l * 3 + 2]);

    Vec3 rij = rj - ri;
    Vec3 rjk = rk - rj;
    Vec3 rkl = rl - rk;

    Vec3 n1 = rij.cross(rjk);
    Vec3 n2 = rjk.cross(rkl);

    double n1_norm = n1.norm();
    double n2_norm = n2.norm();

    if (n1_norm < 1e-10 || n2_norm < 1e-10) {
        partial_energies[idx] = 0.0;
        return;
    }

    // Compute dihedral angle
    double cos_phi = n1.dot(n2) / (n1_norm * n2_norm);
    cos_phi = fmax(-1.0, fmin(1.0, cos_phi));

    double sin_phi = rij.dot(n2) / (rij.norm() * n2_norm);
    double phi = atan2(sin_phi, cos_phi);

    double phi0 = target_dihedrals[idx];
    double force_constant = force_constants[idx];

    // Handle periodic boundary: ensure dphi is in [-pi, pi]
    double dphi = phi - phi0;
    while (dphi > M_PI) dphi -= 2.0 * M_PI;
    while (dphi < -M_PI) dphi += 2.0 * M_PI;

    double energy = 0.5 * force_constant * dphi * dphi;
    partial_energies[idx] = energy;

    // Analytical force calculation for dihedral
    // dE/dphi = k * dphi
    double dE_dphi = force_constant * dphi;

    // Get bond lengths
    double r_ij = rij.norm();
    double r_jk = rjk.norm();
    double r_kl = rkl.norm();

    if (r_jk < 1e-10) return; // Avoid division by zero

    // Compute forces using analytical derivatives
    // Based on standard dihedral force formulation
    Vec3 n1_unit = n1 * (1.0 / n1_norm);
    Vec3 n2_unit = n2 * (1.0 / n2_norm);

    // Force on atom i
    Vec3 fi = n1_unit * (dE_dphi / (n1_norm * r_jk));

    // Force on atom l
    Vec3 fl = n2_unit * (-dE_dphi / (n2_norm * r_jk));

    // Forces on atoms j and k (more complex)
    double dot_ij_jk = rij.dot(rjk) / (r_jk * r_jk);
    double dot_kl_jk = rkl.dot(rjk) / (r_jk * r_jk);

    Vec3 fj = fi * (-1.0 + dot_ij_jk) + fl * (-dot_kl_jk);
    Vec3 fk = fi * (-dot_ij_jk) + fl * (1.0 - dot_kl_jk);

    // Add forces atomically
    for (int d = 0; d < 3; ++d) {
        double f_i = (d == 0) ? fi.x : (d == 1) ? fi.y : fi.z;
        double f_j = (d == 0) ? fj.x : (d == 1) ? fj.y : fj.z;
        double f_k = (d == 0) ? fk.x : (d == 1) ? fk.y : fk.z;
        double f_l = (d == 0) ? fl.x : (d == 1) ? fl.y : fl.z;

        atomicAddDouble(&forces[i * 3 + d], f_i);
        atomicAddDouble(&forces[j * 3 + d], f_j);
        atomicAddDouble(&forces[k * 3 + d], f_k);
        atomicAddDouble(&forces[l * 3 + d], f_l);
    }
}

void harmonic_dihedral_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_dihedrals,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nrestraints * sizeof(double)));

    int nblocks = (nrestraints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    harmonic_dihedral_restraint_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_indices, target_dihedrals, force_constants,
        nrestraints, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[nrestraints];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nrestraints * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nrestraints; ++i) {
        total_energy += h_partial_energies[i];
    }
    CUDA_CHECK(cudaMemcpy(energy, &total_energy, sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Spherical boundary restraint kernel
__global__ void spherical_boundary_restraint_kernel(
    const double* coordinates,
    const int* atom_indices,
    const double* center,
    double radius,
    double force_constant,
    int nrestraints,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrestraints) return;

    int atom_idx = atom_indices[idx];

    Vec3 r(coordinates[atom_idx * 3 + 0], coordinates[atom_idx * 3 + 1], coordinates[atom_idx * 3 + 2]);
    Vec3 c(center[0], center[1], center[2]);

    Vec3 dr = r - c;
    double dist = dr.norm();

    double violation = fmax(0.0, dist - radius);
    double energy = 0.5 * force_constant * violation * violation;
    partial_energies[idx] = energy;

    if (violation > 1e-10 && dist > 1e-10) {
        Vec3 force_dir = dr * (-force_constant * violation / dist);

        for (int d = 0; d < 3; ++d) {
            double f = (d == 0) ? force_dir.x : (d == 1) ? force_dir.y : force_dir.z;
            atomicAddDouble(&forces[atom_idx * 3 + d], f);
        }
    }
}

void spherical_boundary_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* center,
    double radius,
    double force_constant,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nrestraints * sizeof(double)));

    int nblocks = (nrestraints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    spherical_boundary_restraint_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_indices, center, radius, force_constant,
        nrestraints, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[nrestraints];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nrestraints * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nrestraints; ++i) {
        total_energy += h_partial_energies[i];
    }
    CUDA_CHECK(cudaMemcpy(energy, &total_energy, sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// RMSD restraint (stub - complex implementation)
// Backside attack restraint kernel
// Combines angle and distance restraints for SN2 reactions
__global__ void backside_attack_restraint_kernel(
    const double* coordinates,
    const int* restraint_indices,
    const double* target_angles,
    const double* angle_force_constants,
    const double* target_distances,
    const double* distance_force_constants,
    int nrestraints,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrestraints) return;

    // Get atom indices (nucleophile, carbon, leaving_group)
    int nu = restraint_indices[idx * 3 + 0];
    int carbon = restraint_indices[idx * 3 + 1];
    int lg = restraint_indices[idx * 3 + 2];

    Vec3 r_nu(coordinates[nu * 3 + 0], coordinates[nu * 3 + 1], coordinates[nu * 3 + 2]);
    Vec3 r_c(coordinates[carbon * 3 + 0], coordinates[carbon * 3 + 1], coordinates[carbon * 3 + 2]);
    Vec3 r_lg(coordinates[lg * 3 + 0], coordinates[lg * 3 + 1], coordinates[lg * 3 + 2]);

    double energy = 0.0;

    // --- Angle Restraint (Nu-C-LG) ---
    Vec3 r_nu_c = r_nu - r_c;
    Vec3 r_lg_c = r_lg - r_c;

    double d_nu_c = r_nu_c.norm();
    double d_lg_c = r_lg_c.norm();

    if (d_nu_c > 1e-10 && d_lg_c > 1e-10) {
        // Compute angle
        double cos_theta = r_nu_c.dot(r_lg_c) / (d_nu_c * d_lg_c);
        cos_theta = fmax(-1.0, fmin(1.0, cos_theta));
        double theta = acos(cos_theta);

        double theta0 = target_angles[idx];
        double k_angle = angle_force_constants[idx];

        // Angle energy
        double dtheta = theta - theta0;
        energy += 0.5 * k_angle * dtheta * dtheta;

        // Angle forces
        if (fabs(sin(theta)) > 1e-10) {
            double dE_dtheta = k_angle * dtheta;

            // Derivatives of angle w.r.t. positions
            Vec3 n_nu_c = r_nu_c * (1.0 / d_nu_c);
            Vec3 n_lg_c = r_lg_c * (1.0 / d_lg_c);

            Vec3 dtheta_dr_nu = (n_lg_c - n_nu_c * cos_theta) * (1.0 / (d_nu_c * sin(theta)));
            Vec3 dtheta_dr_lg = (n_nu_c - n_lg_c * cos_theta) * (1.0 / (d_lg_c * sin(theta)));
            Vec3 dtheta_dr_c = (dtheta_dr_nu + dtheta_dr_lg) * (-1.0);

            Vec3 f_nu = dtheta_dr_nu * (-dE_dtheta);
            Vec3 f_lg = dtheta_dr_lg * (-dE_dtheta);
            Vec3 f_c = dtheta_dr_c * (-dE_dtheta);

            // Accumulate forces
            for (int d = 0; d < 3; ++d) {
                double val_nu = (d == 0) ? f_nu.x : (d == 1) ? f_nu.y : f_nu.z;
                double val_lg = (d == 0) ? f_lg.x : (d == 1) ? f_lg.y : f_lg.z;
                double val_c = (d == 0) ? f_c.x : (d == 1) ? f_c.y : f_c.z;

                atomicAddDouble(&forces[nu * 3 + d], val_nu);
                atomicAddDouble(&forces[lg * 3 + d], val_lg);
                atomicAddDouble(&forces[carbon * 3 + d], val_c);
            }
        }
    }

    // --- Distance Restraint (Nu-C) ---
    double target_dist = target_distances[idx];
    if (target_dist > 0.0 && d_nu_c > 1e-10) {
        double k_dist = distance_force_constants[idx];

        // Distance energy
        double dr = d_nu_c - target_dist;
        energy += 0.5 * k_dist * dr * dr;

        // Distance forces (negative of energy gradient to pull atoms together)
        double force_mag = -k_dist * dr;
        Vec3 force_dir = r_nu_c * (1.0 / d_nu_c);
        Vec3 f = force_dir * force_mag;

        for (int d = 0; d < 3; ++d) {
            double val = (d == 0) ? f.x : (d == 1) ? f.y : f.z;
            atomicAddDouble(&forces[nu * 3 + d], val);
            atomicAddDouble(&forces[carbon * 3 + d], -val);
        }
    }

    partial_energies[idx] = energy;
}

void backside_attack_restraint(
    const double* coordinates,
    const int* restraint_indices,
    const double* target_angles,
    const double* angle_force_constants,
    const double* target_distances,
    const double* distance_force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nrestraints * sizeof(double)));

    int nblocks = (nrestraints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    backside_attack_restraint_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, restraint_indices, target_angles, angle_force_constants,
        target_distances, distance_force_constants, nrestraints,
        d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    // Reduce energies
    double* h_partial_energies = new double[nrestraints];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nrestraints * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nrestraints; ++i) {
        total_energy += h_partial_energies[i];
    }
    CUDA_CHECK(cudaMemcpy(energy, &total_energy, sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void rmsd_restraint(
    const double* coordinates,
    const double* reference,
    const int* atom_indices,
    const double* masses,
    double target_rmsd,
    double force_constant,
    int natoms,
    int nrestraints,
    bool use_mass_weighting,
    double* energy,
    double* forces
) {
    // RMSD restraint requires Kabsch alignment which is complex
    // For now, this is a placeholder
    // Full implementation would require:
    // 1. Compute center of mass
    // 2. Kabsch algorithm for optimal alignment
    // 3. Compute RMSD
    // 4. Compute derivatives (chain rule through rotation)
    double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(energy, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda
} // namespace fennol
