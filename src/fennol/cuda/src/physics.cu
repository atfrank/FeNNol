#include "../include/physics.cuh"
#include <cmath>

namespace fennol {
namespace cuda {

// Lennard-Jones 12-6 kernel
__global__ void lennard_jones_kernel(
    const double* coordinates,
    const int* atom_pairs,
    const double* epsilons,
    const double* sigmas,
    int npairs,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npairs) return;

    int i = atom_pairs[idx * 2 + 0];
    int j = atom_pairs[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    if (r < 1e-10) {
        partial_energies[idx] = 0.0;
        return;
    }

    double epsilon = epsilons[idx];
    double sigma = sigmas[idx];

    // LJ potential: 4*eps * ((sig/r)^12 - (sig/r)^6)
    double sig_r = sigma / r;
    double sig_r6 = sig_r * sig_r * sig_r * sig_r * sig_r * sig_r;
    double sig_r12 = sig_r6 * sig_r6;

    double energy = 4.0 * epsilon * (sig_r12 - sig_r6);
    partial_energies[idx] = energy;

    // Force: dE/dr = 24*eps * (2*sig^12/r^13 - sig^6/r^7)
    double force_mag = 24.0 * epsilon * (2.0 * sig_r12 - sig_r6) / r;
    Vec3 force_vec = rij * (force_mag / r);

    for (int d = 0; d < 3; ++d) {
        double f = (d == 0) ? force_vec.x : (d == 1) ? force_vec.y : force_vec.z;
        atomicAddDouble(&forces[i * 3 + d], f);
        atomicAddDouble(&forces[j * 3 + d], -f);
    }
}

void lennard_jones_pairwise(
    const double* coordinates,
    const int* atom_pairs,
    const double* epsilons,
    const double* sigmas,
    int natoms,
    int npairs,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, npairs * sizeof(double)));

    int nblocks = (npairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    lennard_jones_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_pairs, epsilons, sigmas, npairs, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    // Reduce energies
    double* h_partial_energies = new double[npairs];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          npairs * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < npairs; ++i) {
        total_energy += h_partial_energies[i];
    }
    *energy = total_energy;

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Coulomb electrostatics kernel
__global__ void coulomb_kernel(
    const double* coordinates,
    const double* charges,
    const int* atom_pairs,
    int npairs,
    double coulomb_constant,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npairs) return;

    int i = atom_pairs[idx * 2 + 0];
    int j = atom_pairs[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    if (r < 1e-10) {
        partial_energies[idx] = 0.0;
        return;
    }

    double qi = charges[i];
    double qj = charges[j];

    // Coulomb energy: k_e * q_i * q_j / r
    double energy = coulomb_constant * qi * qj / r;
    partial_energies[idx] = energy;

    // Force: F = k_e * q_i * q_j / r^2 * (r_ij / r)
    double force_mag = coulomb_constant * qi * qj / (r * r);
    Vec3 force_vec = rij * (force_mag / r);

    for (int d = 0; d < 3; ++d) {
        double f = (d == 0) ? force_vec.x : (d == 1) ? force_vec.y : force_vec.z;
        atomicAddDouble(&forces[i * 3 + d], f);
        atomicAddDouble(&forces[j * 3 + d], -f);
    }
}

void coulomb_direct(
    const double* coordinates,
    const double* charges,
    const int* atom_pairs,
    int natoms,
    int npairs,
    double coulomb_constant,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, npairs * sizeof(double)));

    int nblocks = (npairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    coulomb_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, charges, atom_pairs, npairs, coulomb_constant,
        d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[npairs];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          npairs * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < npairs; ++i) {
        total_energy += h_partial_energies[i];
    }
    *energy = total_energy;

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ZBL screening function
__device__ double zbl_screening(double x) {
    // ZBL universal screening function
    // phi(x) = 0.1818*exp(-3.2*x) + 0.5099*exp(-0.9423*x) +
    //          0.2802*exp(-0.4029*x) + 0.02817*exp(-0.2016*x)
    return 0.1818 * exp(-3.2 * x) +
           0.5099 * exp(-0.9423 * x) +
           0.2802 * exp(-0.4029 * x) +
           0.02817 * exp(-0.2016 * x);
}

__device__ double zbl_screening_derivative(double x) {
    // Derivative of ZBL screening function
    return -0.1818 * 3.2 * exp(-3.2 * x) -
           0.5099 * 0.9423 * exp(-0.9423 * x) -
           0.2802 * 0.4029 * exp(-0.4029 * x) -
           0.02817 * 0.2016 * exp(-0.2016 * x);
}

// ZBL repulsion kernel
__global__ void zbl_kernel(
    const double* coordinates,
    const int* atomic_numbers,
    const int* atom_pairs,
    int npairs,
    double cutoff,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npairs) return;

    int i = atom_pairs[idx * 2 + 0];
    int j = atom_pairs[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    if (r < 1e-10 || r > cutoff) {
        partial_energies[idx] = 0.0;
        return;
    }

    int Zi = atomic_numbers[i];
    int Zj = atomic_numbers[j];

    // ZBL screening length (Angstroms)
    double a0 = 0.529; // Bohr radius in Angstroms
    double au = 0.8854 * a0 / (pow(Zi, 0.23) + pow(Zj, 0.23));

    // ZBL potential
    double ke = 14.3996; // eV*Angstrom (Coulomb constant)
    double x = r / au;
    double phi = zbl_screening(x);
    double dphi_dx = zbl_screening_derivative(x);

    double energy = ke * Zi * Zj * phi / r;
    partial_energies[idx] = energy;

    // Force: F = -dE/dr
    // dE/dr = ke * Zi * Zj * (dphi/dx/(au*r) - phi/r^2)
    double dE_dr = ke * Zi * Zj * (dphi_dx / (au * r) - phi / (r * r));
    Vec3 force_vec = rij * (-dE_dr / r);

    for (int d = 0; d < 3; ++d) {
        double f = (d == 0) ? force_vec.x : (d == 1) ? force_vec.y : force_vec.z;
        atomicAddDouble(&forces[i * 3 + d], f);
        atomicAddDouble(&forces[j * 3 + d], -f);
    }
}

void zbl_repulsion(
    const double* coordinates,
    const int* atomic_numbers,
    const int* atom_pairs,
    int natoms,
    int npairs,
    double cutoff,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, npairs * sizeof(double)));

    int nblocks = (npairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    zbl_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atomic_numbers, atom_pairs, npairs, cutoff,
        d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[npairs];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          npairs * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < npairs; ++i) {
        total_energy += h_partial_energies[i];
    }
    *energy = total_energy;

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// NLH repulsion kernel
// Based on Nordlund-Lehtola-Hobler pair-specific repulsion model
// Uses three exponential terms with pair-specific coefficients
__global__ void nlh_kernel(
    const double* coordinates,
    const int* atomic_numbers,
    const int* atom_pairs,
    const double* pair_coefficients,
    int npairs,
    double cutoff,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npairs) return;

    int i = atom_pairs[idx * 2 + 0];
    int j = atom_pairs[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    if (r < 1e-10 || r > cutoff) {
        partial_energies[idx] = 0.0;
        return;
    }

    int Zi = atomic_numbers[i];
    int Zj = atomic_numbers[j];

    // Get pair-specific coefficients (a1, b1, a2, b2, a3, b3)
    double a1 = pair_coefficients[idx * 6 + 0];
    double b1 = pair_coefficients[idx * 6 + 1];
    double a2 = pair_coefficients[idx * 6 + 2];
    double b2 = pair_coefficients[idx * 6 + 3];
    double a3 = pair_coefficients[idx * 6 + 4];
    double b3 = pair_coefficients[idx * 6 + 5];

    // NLH potential: E = (Z_i * Z_j * k_e / r) * phi(r)
    // phi(r) = a1*exp(-b1*r) + a2*exp(-b2*r) + a3*exp(-b3*r)
    double ke = 14.3996; // eV*Angstrom (Coulomb constant)

    double exp1 = exp(-b1 * r);
    double exp2 = exp(-b2 * r);
    double exp3 = exp(-b3 * r);

    double phi = a1 * exp1 + a2 * exp2 + a3 * exp3;
    double dphi_dr = -(a1 * b1 * exp1 + a2 * b2 * exp2 + a3 * b3 * exp3);

    double energy = ke * Zi * Zj * phi / r;
    partial_energies[idx] = energy;

    // Force: F = -dE/dr
    // dE/dr = ke * Zi * Zj * d/dr(phi/r) = ke * Zi * Zj * (dphi/dr/r - phi/r^2)
    double dE_dr = ke * Zi * Zj * (dphi_dr / r - phi / (r * r));
    Vec3 force_vec = rij * (-dE_dr / r);

    for (int d = 0; d < 3; ++d) {
        double f = (d == 0) ? force_vec.x : (d == 1) ? force_vec.y : force_vec.z;
        atomicAddDouble(&forces[i * 3 + d], f);
        atomicAddDouble(&forces[j * 3 + d], -f);
    }
}

void nlh_repulsion(
    const double* coordinates,
    const int* atomic_numbers,
    const int* atom_pairs,
    const double* pair_coefficients,
    int natoms,
    int npairs,
    double cutoff,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, npairs * sizeof(double)));

    int nblocks = (npairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    nlh_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atomic_numbers, atom_pairs, pair_coefficients, npairs, cutoff,
        d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[npairs];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          npairs * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < npairs; ++i) {
        total_energy += h_partial_energies[i];
    }
    *energy = total_energy;

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Dispersion C6 kernel
__global__ void dispersion_kernel(
    const double* coordinates,
    const int* atom_pairs,
    const double* c6_coefficients,
    int npairs,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npairs) return;

    int i = atom_pairs[idx * 2 + 0];
    int j = atom_pairs[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    if (r < 1e-10) {
        partial_energies[idx] = 0.0;
        return;
    }

    double c6 = c6_coefficients[idx];

    // Dispersion energy: -C6 / r^6
    double r6 = r * r * r * r * r * r;
    double energy = -c6 / r6;
    partial_energies[idx] = energy;

    // Force: dE/dr = -6 * C6 / r^7
    double force_mag = -6.0 * c6 / (r6 * r);
    Vec3 force_vec = rij * (force_mag / r);

    for (int d = 0; d < 3; ++d) {
        double f = (d == 0) ? force_vec.x : (d == 1) ? force_vec.y : force_vec.z;
        atomicAddDouble(&forces[i * 3 + d], f);
        atomicAddDouble(&forces[j * 3 + d], -f);
    }
}

void dispersion_c6(
    const double* coordinates,
    const int* atom_pairs,
    const double* c6_coefficients,
    int natoms,
    int npairs,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, npairs * sizeof(double)));

    int nblocks = (npairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dispersion_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, atom_pairs, c6_coefficients, npairs,
        d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[npairs];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          npairs * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < npairs; ++i) {
        total_energy += h_partial_energies[i];
    }
    *energy = total_energy;

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Harmonic bonds (same as distance restraints but for bonded topology)
__global__ void harmonic_bonds_kernel(
    const double* coordinates,
    const int* bond_indices,
    const double* equilibrium_lengths,
    const double* force_constants,
    int nbonds,
    double* partial_energies,
    double* forces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbonds) return;

    int i = bond_indices[idx * 2 + 0];
    int j = bond_indices[idx * 2 + 1];

    Vec3 ri(coordinates[i * 3 + 0], coordinates[i * 3 + 1], coordinates[i * 3 + 2]);
    Vec3 rj(coordinates[j * 3 + 0], coordinates[j * 3 + 1], coordinates[j * 3 + 2]);

    Vec3 rij = rj - ri;
    double r = rij.norm();

    double r0 = equilibrium_lengths[idx];
    double k = force_constants[idx];
    double dr = r - r0;

    double energy = 0.5 * k * dr * dr;
    partial_energies[idx] = energy;

    if (r > 1e-10) {
        Vec3 force_dir = rij * (k * dr / r);

        for (int d = 0; d < 3; ++d) {
            double f = (d == 0) ? force_dir.x : (d == 1) ? force_dir.y : force_dir.z;
            atomicAddDouble(&forces[i * 3 + d], f);
            atomicAddDouble(&forces[j * 3 + d], -f);
        }
    }
}

void harmonic_bonds(
    const double* coordinates,
    const int* bond_indices,
    const double* equilibrium_lengths,
    const double* force_constants,
    int natoms,
    int nbonds,
    double* energy,
    double* forces
) {
    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nbonds * sizeof(double)));

    int nblocks = (nbonds + BLOCK_SIZE - 1) / BLOCK_SIZE;

    harmonic_bonds_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, bond_indices, equilibrium_lengths, force_constants,
        nbonds, d_partial_energies, forces
    );
    CUDA_CHECK(cudaGetLastError());

    double* h_partial_energies = new double[nbonds];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nbonds * sizeof(double), cudaMemcpyDeviceToHost));

    double total_energy = 0.0;
    for (int i = 0; i < nbonds; ++i) {
        total_energy += h_partial_energies[i];
    }
    *energy = total_energy;

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Harmonic angles (same as angle restraints but for bonded topology)
// Implementation same as in restraints.cu - reuse that code

void harmonic_angles(
    const double* coordinates,
    const int* angle_indices,
    const double* equilibrium_angles,
    const double* force_constants,
    int natoms,
    int nangles,
    double* energy,
    double* forces
) {
    // This is essentially the same as harmonic_angle_restraint
    // We can reuse that kernel by wrapping it
    // For now, placeholder - in production would share implementation
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda
} // namespace fennol
