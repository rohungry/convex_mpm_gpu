// multibody/gpu_mpm/cuda_mpm_hvp_kernels.cuh
//
// Matrix-free Hessian-vector product for the convex contact solve
// (arXiv:2503.05046), written against the ACTUAL fork kernels in
// cuda_mpm_kernels.cuh. Key facts established from that source:
//
//  * Per-contact world Hessian  H_c^W = R_WC * lc_Hess_C * R_CW, exactly as in
//    contact_particle_to_grid_kernel (their comment: "J^T Hess J"). lc_Hess_C
//    comes from the EXISTING compute_contact_grad_and_hess() -- reused, not
//    re-derived.
//  * grid<->contact transfer is the quadratic B-spline triple loop with scalar
//    weights w_ic and NO rotation in the loop (the rotation lives inside
//    H_c^W). So  (J p)_c = sum_i w_ic * grid_P[node_i]  (a plain Vec3), and
//    scatter is  grid_Hp[node_i] += w_ic * (H_c^W (J p)_c).
//  * Full operator  H p = M p + sum_c J_c^T H_c^W (J_c p). The i!=j cross terms
//    are the off-diagonal node coupling that the block-Jacobi solve discards;
//    this operator keeps them.
//
// H_c^W (9 floats/contact) is cached in contact_Hess, precomputed once per
// linearization point (depends on the iterate velocity, frozen across inner CG).

#pragma once

#include "multibody/gpu_mpm/settings.h"
#include "multibody/gpu_mpm/math_tools.cuh"
// cuda_mpm_kernels.cuh (included before this file in cuda_mpm_solver.cu) provides:
//   GridConfig<T>, cell_index(), make_from_one_unit_vector(),
//   compute_contact_grad_and_hess(), matmul<>, transpose<>.

namespace drake {
namespace multibody {
namespace gmpm {

// ---------------------------------------------------------------------------
// (0) Precompute H_c^W = R_WC * lc_Hess_C * R_CW per contact, once per
//     linearization. Mirrors the rotation in contact_particle_to_grid_kernel
//     and reuses compute_contact_grad_and_hess(). contact_Hess layout:
//     row-major 3x3 per contact (9 T), same convention as g_Hess.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void precompute_contact_world_hessian_kernel(
    const size_t n_contacts,
    const T* contact_vel,        // current iterate v_p at the contact (world)
    const T* velocities,         // v_p^n (previous step), indexed by contact_mpm_id
    const uint32_t* contact_mpm_id,
    const T* contact_dist,
    const T* contact_normal,
    const T* contact_rigid_v,
    T* contact_Hess,             // out: 9 T per contact (H_c^W, row-major)
    const T dt, const T friction_mu, const T stiffness,
    const T epsv, const T damping) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= n_contacts) return;

  const T* particle_vn = &velocities[contact_mpm_id[idx] * 3];  // v0 = v_p^n
  const T* particle_v  = &contact_vel[idx * 3];                 // v_next iterate

  T nhat_W[3] = {contact_normal[idx * 3 + 0], contact_normal[idx * 3 + 1],
                 contact_normal[idx * 3 + 2]};
  T phi0 = -contact_dist[idx];

  T vn_rel_W[3] = {particle_vn[0] - contact_rigid_v[idx * 3 + 0],
                   particle_vn[1] - contact_rigid_v[idx * 3 + 1],
                   particle_vn[2] - contact_rigid_v[idx * 3 + 2]};
  T v_rel_W[3] = {particle_v[0] - contact_rigid_v[idx * 3 + 0],
                  particle_v[1] - contact_rigid_v[idx * 3 + 1],
                  particle_v[2] - contact_rigid_v[idx * 3 + 2]};

  constexpr int kZAxis = 2;
  T R_WC[9], R_CW[9];
  make_from_one_unit_vector(nhat_W, kZAxis, R_WC);
  transpose<3, 3, T>(R_WC, R_CW);

  T vn_C[3], v_next_C[3];
  matmul<3, 3, 1, T>(R_CW, vn_rel_W, vn_C);
  matmul<3, 3, 1, T>(R_CW, v_rel_W, v_next_C);

  T lc_Hess_C[9], lc_Grad_C_unused[3];
  compute_contact_grad_and_hess(phi0, dt, stiffness, epsv, damping, friction_mu,
                                vn_C, v_next_C, lc_Hess_C, lc_Grad_C_unused);

  // H_c^W = R_WC * lc_Hess_C * R_CW  (identical to the assembly kernel)
  T tmp[9], Hc_W[9];
  matmul<3, 3, 3, T>(R_WC, lc_Hess_C, tmp);
  matmul<3, 3, 3, T>(tmp, R_CW, Hc_W);
  #pragma unroll
  for (int i = 0; i < 9; ++i) contact_Hess[idx * 9 + i] = Hc_W[i];
}

// ---------------------------------------------------------------------------
// (a) gather + local apply:  contact_scratch_c = H_c^W * (J p)_c
//     (J p)_c = sum_i w_ic * grid_P[node_i]   (scalar-weight gather; same
//     B-spline weights as grid_to_particle_kernel / the contact P2G).
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_DIM>
__global__ void hv_gather_apply_kernel(
    const GridConfig<T> gconf, const size_t n_contacts,
    const T* contact_pos,
    const T* contact_Hess,   // H_c^W per contact (9 T)
    const T* grid_P,         // input p on the grid (3 T per cell)
    T* contact_scratch) {    // out: 3 T per contact
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ T weights[BLOCK_DIM][3][3];
  if (idx >= n_contacts) return;

  uint32_t base[3] = {
      static_cast<uint32_t>(contact_pos[idx * 3 + 0] * gconf.G_DX_INV - T(0.5)),
      static_cast<uint32_t>(contact_pos[idx * 3 + 1] * gconf.G_DX_INV - T(0.5)),
      static_cast<uint32_t>(contact_pos[idx * 3 + 2] * gconf.G_DX_INV - T(0.5))};
  T fx[3] = {contact_pos[idx * 3 + 0] * gconf.G_DX_INV - static_cast<T>(base[0]),
             contact_pos[idx * 3 + 1] * gconf.G_DX_INV - static_cast<T>(base[1]),
             contact_pos[idx * 3 + 2] * gconf.G_DX_INV - static_cast<T>(base[2])};
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
    weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
    weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
  }

  T Jp[3] = {0, 0, 0};
  #pragma unroll
  for (int i = 0; i < 3; ++i)
    #pragma unroll
    for (int j = 0; j < 3; ++j)
      #pragma unroll
      for (int k = 0; k < 3; ++k) {
        const uint32_t cell = cell_index(gconf, base[0] + i, base[1] + j, base[2] + k);
        const T w = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] *
                    weights[threadIdx.x][k][2];
        const T* gp = &grid_P[cell * 3];
        Jp[0] += w * gp[0]; Jp[1] += w * gp[1]; Jp[2] += w * gp[2];
      }

  const T* H = &contact_Hess[idx * 9];
  contact_scratch[idx * 3 + 0] = H[0] * Jp[0] + H[1] * Jp[1] + H[2] * Jp[2];
  contact_scratch[idx * 3 + 1] = H[3] * Jp[0] + H[4] * Jp[1] + H[5] * Jp[2];
  contact_scratch[idx * 3 + 2] = H[6] * Jp[0] + H[7] * Jp[1] + H[8] * Jp[2];
}

// ---------------------------------------------------------------------------
// (b) scatter (the J^T):  grid_Hp[node_i] += w_ic * contact_scratch_c
//     Same warp-reduction + boundary pattern as contact_particle_to_grid_kernel,
//     so atomics land once per (cell, warp-run) and J^T == transpose(J).
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_DIM>
__global__ void hv_scatter_kernel(
    const GridConfig<T> gconf, const size_t n_contacts,
    const T* contact_pos,
    const uint32_t* grid_index,     // contact_sort_keys
    const T* contact_scratch,       // 3 T per contact
    T* grid_Hp) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ T weights[BLOCK_DIM][3][3];

  int laneid = threadIdx.x & 0x1f;
  int cellid = -1;
  bool boundary;
  if (idx < n_contacts) {
    cellid = grid_index[idx];
    boundary = (laneid == 0) || cellid != grid_index[idx - 1];
  } else {
    boundary = true;
  }
  uint32_t mark = __ballot_sync(0xFFFFFFFF, boundary);
  mark = __brev(mark);
  unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
  mark = interval;
  #pragma unroll
  for (int iter = 1; iter & 0x1f; iter <<= 1) {
    int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
    mark = tmp > mark ? tmp : mark;
  }
  mark = __shfl_sync(0xFFFFFFFF, mark, 0);

  if (idx >= n_contacts) return;

  uint32_t base[3] = {
      static_cast<uint32_t>(contact_pos[idx * 3 + 0] * gconf.G_DX_INV - T(0.5)),
      static_cast<uint32_t>(contact_pos[idx * 3 + 1] * gconf.G_DX_INV - T(0.5)),
      static_cast<uint32_t>(contact_pos[idx * 3 + 2] * gconf.G_DX_INV - T(0.5))};
  T fx[3] = {contact_pos[idx * 3 + 0] * gconf.G_DX_INV - static_cast<T>(base[0]),
             contact_pos[idx * 3 + 1] * gconf.G_DX_INV - static_cast<T>(base[1]),
             contact_pos[idx * 3 + 2] * gconf.G_DX_INV - static_cast<T>(base[2])};
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    weights[threadIdx.x][0][i] = T(0.5) * (T(1.5) - fx[i]) * (T(1.5) - fx[i]);
    weights[threadIdx.x][1][i] = T(0.75) - (fx[i] - T(1.0)) * (fx[i] - T(1.0));
    weights[threadIdx.x][2][i] = T(0.5) * (fx[i] - T(0.5)) * (fx[i] - T(0.5));
  }

  const T* s = &contact_scratch[idx * 3];
  #pragma unroll
  for (int i = 0; i < 3; ++i)
    #pragma unroll
    for (int j = 0; j < 3; ++j)
      #pragma unroll
      for (int k = 0; k < 3; ++k) {
        T w = weights[threadIdx.x][i][0] * weights[threadIdx.x][j][1] *
              weights[threadIdx.x][k][2];
        T val[3] = {w * s[0], w * s[1], w * s[2]};
        for (int iter = 1; iter <= mark; iter <<= 1) {
          T tmp[3];
          #pragma unroll
          for (int ii = 0; ii < 3; ++ii)
            tmp[ii] = __shfl_down_sync(0xFFFFFFFF, val[ii], iter);
          if (interval >= iter) {
            #pragma unroll
            for (int ii = 0; ii < 3; ++ii) val[ii] += tmp[ii];
          }
        }
        if (boundary) {
          const uint32_t cell = cell_index(gconf, base[0] + i, base[1] + j, base[2] + k);
          atomicAdd(&grid_Hp[cell * 3 + 0], val[0]);
          atomicAdd(&grid_Hp[cell * 3 + 1], val[1]);
          atomicAdd(&grid_Hp[cell * 3 + 2], val[2]);
        }
      }
}

// ---------------------------------------------------------------------------
// (c) clean + mass term over touched cells (same cell-index decode as
//     clean_grid_contact_kernel / grid_contact_3x3_parallel_solving_kernel).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void hv_clean_kernel(
    const GridConfig<T> gconf, const uint32_t touched_cells_cnt,
    const uint32_t* g_touched_ids, T* grid_Hp) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= touched_cells_cnt) return;
  uint32_t block_idx = g_touched_ids[idx >> (gconf.G_BLOCK_BITS * 3)];
  uint32_t cell_idx = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
  grid_Hp[cell_idx * 3 + 0] = 0;
  grid_Hp[cell_idx * 3 + 1] = 0;
  grid_Hp[cell_idx * 3 + 2] = 0;
}

template <typename T>
__global__ void hv_mass_add_kernel(
    const GridConfig<T> gconf, const uint32_t touched_cells_cnt,
    const uint32_t* g_touched_ids, const T* g_masses,
    const T* grid_P, T* grid_Hp) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= touched_cells_cnt) return;
  uint32_t block_idx = g_touched_ids[idx >> (gconf.G_BLOCK_BITS * 3)];
  uint32_t cell_idx = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
  const T m = g_masses[cell_idx];
  if (m > T(0.)) {
    grid_Hp[cell_idx * 3 + 0] += m * grid_P[cell_idx * 3 + 0];
    grid_Hp[cell_idx * 3 + 1] += m * grid_P[cell_idx * 3 + 1];
    grid_Hp[cell_idx * 3 + 2] += m * grid_P[cell_idx * 3 + 2];
  }
}

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake
