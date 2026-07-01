// multibody/gpu_mpm/cuda_mpm_hvp.inc.cu
//
// Host-side driver for the matrix-free Hessian-vector product + self-tests.
// #include this at the BOTTOM of cuda_mpm_solver.cu, before the explicit
// instantiation `template class GpuMpmSolver<config::GpuT>;`, so it shares the
// template, CUDA_SAFE_CALL, and the kernels already in scope.
//
// Provides on GpuMpmSolver<T>:
//   PrecomputeContactHessian(state)        -- cache H_c^W, once per linearization
//   ApplyHessian(state, p_dev, Hp_dev)     -- Hp = H p, matrix-free, inner-CG core
//   AssembleContactGradient(state, g_dev)  -- g = M(v - v*) + J^T dl_c/dvc at current v
//   RunHvpSelfTests(state)                 -- FD (against the SAME gradient) + symmetry + PD
//
// IMPORTANT correctness note (see INTEGRATION.md "lagged friction"): the FD test
// differences AssembleContactGradient, NOT the line-search energy `lc`. The
// friction Hessian is the lagged/Gauss-Newton model (yn0 frozen at v^n), so it
// matches the gradient's linearization, not d^2(energy). Differencing the energy
// will spuriously "fail" -- that is expected, not a bug.

//#include <random>  // fill_random_touched_impl
#include "multibody/gpu_mpm/cuda_mpm_hvp_kernels.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

// namespace hvp_detail {
constexpr int kBlk = config::DEFAULT_CUDA_BLOCK_SIZE;
inline int grid_for(int n) { return (n + kBlk - 1) / kBlk; }

// reduction: sum over touched cells of a . b  (3-vectors per cell)
template <typename T>
__global__ void dot_touched_kernel(const GridConfig<T> gconf, uint32_t tc,
                                   const uint32_t* ids, const T* a, const T* b, T* out) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= tc) return;
  uint32_t block_idx = ids[idx >> (gconf.G_BLOCK_BITS * 3)];
  uint32_t cell = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
  const T v = a[cell*3+0]*b[cell*3+0] + a[cell*3+1]*b[cell*3+1] + a[cell*3+2]*b[cell*3+2];
  atomicAdd(out, v);
}

// out[cell] = a[cell] + alpha * d[cell]  over touched cells (others untouched)
template <typename T>
__global__ void axpy_touched_kernel(const GridConfig<T> gconf, uint32_t tc,
                                    const uint32_t* ids, const T* a, const T* d,
                                    T alpha, T* out) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= tc) return;
  uint32_t block_idx = ids[idx >> (gconf.G_BLOCK_BITS * 3)];
  uint32_t cell = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
  out[cell*3+0] = a[cell*3+0] + alpha * d[cell*3+0];
  out[cell*3+1] = a[cell*3+1] + alpha * d[cell*3+1];
  out[cell*3+2] = a[cell*3+2] + alpha * d[cell*3+2];
}

// sum over touched cells of || (gp-gm)*scale - Hp ||^2  and  ||Hp||^2
template <typename T>
__global__ void fd_resid_kernel(const GridConfig<T> gconf, uint32_t tc,
                                const uint32_t* ids, const T* gp, const T* gm,
                                const T* Hp, T scale, T* out_resid_sq, T* out_Hp_sq) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= tc) return;
  uint32_t block_idx = ids[idx >> (gconf.G_BLOCK_BITS * 3)];
  uint32_t cell = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
  T rs = 0, hs = 0;
  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    T fd = (gp[cell*3+c] - gm[cell*3+c]) * scale;
    T r = fd - Hp[cell*3+c];
    rs += r * r;  hs += Hp[cell*3+c] * Hp[cell*3+c];
  }
  atomicAdd(out_resid_sq, rs);
  atomicAdd(out_Hp_sq, hs);
}
// }  // namespace hvp_detail

// Forward declaration (defined below; used by AssembleContactGradient).
template <typename T>
__global__ void mass_residual_add_kernel(
    const GridConfig<T> gconf, const uint32_t tc, const uint32_t* ids,
    const T* g_masses, const T* g_momentum, const T* g_v_star,
    const T* g_Grad, T* g_out);

// Host helper you must provide once (fills a grid Vec3 buffer with random
// values on touched cells, zero elsewhere). A reference implementation:
//
//   template <typename T>
//   void GpuMpmSolver<T>::fill_random_touched(GpuMpmState<T>* s, T* dst) const {
//     // copy touched ids to host, build a host buffer of size 3*G_DOMAIN_VOLUME
//     // zero-initialized, write randn() into the 3 slots of each touched cell,
//     // cudaMemcpy to dst. (Touched-cell decode same as the kernels above.)
//   }
//
// Declared here so RunHvpSelfTests compiles; implement in the solver .cu.
template <typename T> void fill_random_touched_impl(GpuMpmState<T>* s, T* dst);
#define fill_random_touched(s, dst) fill_random_touched_impl((s), (dst))

// ---------------------------------------------------------------------------
template <typename T>
void GpuMpmSolver<T>::PrecomputeContactHessian(GpuMpmState<T>* s, const T& dt) const {
  // using namespace hvp_detail;
  const int nc = s->num_contacts();
  if (nc == 0) return;
  CUDA_SAFE_CALL((precompute_contact_world_hessian_kernel<T><<<grid_for(nc), kBlk>>>(
      nc, s->contact_vel(), s->current_velocities(), s->contact_mpm_id(),
      s->contact_dist(), s->contact_normal(), s->contact_rigid_v(),
      s->contact_Hess(),
      dt, s->config().contact_friction_mu, s->config().contact_stiffness,
      s->config().contact_epsv, s->config().contact_damping)));
}

// Hp = M p + J^T H_c^W (J p). Requires PrecomputeContactHessian first.
template <typename T>
void GpuMpmSolver<T>::ApplyHessian(GpuMpmState<T>* s, const T* p_dev, T* Hp_dev) const {
  // using namespace hvp_detail;
  const int nc = s->num_contacts();
  const uint32_t tc = s->grid_touched_cnt_host() * s->grid_config().G_BLOCK_VOLUME;

  if (tc > 0)
    CUDA_SAFE_CALL((hv_clean_kernel<T><<<grid_for(tc), kBlk>>>(
        s->grid_config(), tc, s->grid_touched_ids(), Hp_dev)));
  if (nc > 0) {
    CUDA_SAFE_CALL((hv_gather_apply_kernel<T, kBlk><<<grid_for(nc), kBlk>>>(
        s->grid_config(), nc, s->contact_pos(), s->contact_Hess(),
        p_dev, s->contact_scratch())));
    CUDA_SAFE_CALL((hv_scatter_kernel<T, kBlk><<<grid_for(nc), kBlk>>>(
        s->grid_config(), nc, s->contact_pos(), s->contact_sort_keys(),
        s->contact_scratch(), Hp_dev)));
  }
  if (tc > 0)
    CUDA_SAFE_CALL((hv_mass_add_kernel<T><<<grid_for(tc), kBlk>>>(
        s->grid_config(), tc, s->grid_touched_ids(), s->grid_masses(),
        p_dev, Hp_dev)));
  this->GpuSync();
}

// g = M (v - v*) + J^T dl_c/dvc, evaluated at the CURRENT grid_momentum (= v).
// Reuses the existing assembly kernels exactly as UpdateContact does, but stops
// before the 3x3 solve. Writes g into g_out_dev (grid Vec3 over touched cells).
//
// Pattern mirrors one UpdateContact inner iteration's gradient stage:
//   1. clean g_Hess/g_Grad/g_Dir over touched cells
//   2. contact_particle_to_grid_kernel -> g_Grad gets J^T dl_c/dvc  (and g_Hess,
//      which we ignore here)
//   3. add M(v - v*) to g_Grad, node-wise, over touched cells
// Step 2 needs contact velocities consistent with v; refresh them with the
// existing contact-G2P (grid_to_particle_kernel<CONTACT_TRANSFER=true>) first.
template <typename T>
void GpuMpmSolver<T>::AssembleContactGradient(GpuMpmState<T>* s, const T& dt,
                                              T* g_out_dev) const {
  // using namespace hvp_detail;
  const int nc = s->num_contacts();
  const uint32_t tc = s->grid_touched_cnt_host() * s->grid_config().G_BLOCK_VOLUME;

  // refresh contact velocities from current grid_momentum (v)
  if (nc > 0)
    CUDA_SAFE_CALL((grid_to_particle_kernel<T, kBlk, /*CONTACT_TRANSFER=*/true>
        <<<grid_for(nc), kBlk>>>(
        s->grid_config(), nc, s->contact_pos(), s->contact_vel(),
        /*affine=*/nullptr, s->grid_masses(), s->grid_momentum(), dt,
        s->config().rpic_damping)));

  // clean Hess/Grad/Dir
  if (tc > 0)
    CUDA_SAFE_CALL((clean_grid_contact_kernel<T><<<grid_for(tc), kBlk>>>(
        s->grid_config(), tc, s->grid_touched_ids(),
        s->grid_Hess(), s->grid_Grad(), s->grid_Dir())));

  // contact gradient -> g_Grad   (we ignore the g_Hess it also writes)
  if (nc > 0)
    CUDA_SAFE_CALL((contact_particle_to_grid_kernel<T, 32><<<grid_for(nc), 32>>>(
        s->grid_config(), nc, s->contact_pos(), s->contact_vel(),
        s->current_velocities(), s->contact_mpm_id(), s->contact_dist(),
        s->contact_normal(), s->contact_rigid_v(), s->contact_sort_keys(),
        s->grid_Hess(), s->grid_Grad(), dt,
        s->config().contact_friction_mu, s->config().contact_stiffness,
        s->config().contact_epsv, s->config().contact_damping)));

  // g_out = g_Grad + M (v - v*)   over touched cells
  if (tc > 0)
    CUDA_SAFE_CALL((mass_residual_add_kernel<T><<<grid_for(tc), kBlk>>>(
        s->grid_config(), tc, s->grid_touched_ids(), s->grid_masses(),
        s->grid_momentum(), s->grid_v_star(), s->grid_Grad(), g_out_dev)));
  this->GpuSync();
}

// g_out[cell] = g_Grad[cell] + m (v[cell] - v*[cell])  over touched cells.
template <typename T>
__global__ void mass_residual_add_kernel(
    const GridConfig<T> gconf, const uint32_t tc, const uint32_t* ids,
    const T* g_masses, const T* g_momentum, const T* g_v_star,
    const T* g_Grad, T* g_out) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= tc) return;
  uint32_t block_idx = ids[idx >> (gconf.G_BLOCK_BITS * 3)];
  uint32_t cell = (block_idx << (gconf.G_BLOCK_BITS * 3)) | (idx & gconf.G_BLOCK_VOLUME_MASK);
  const T m = g_masses[cell];
  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    T mr = (m > T(0.)) ? m * (g_momentum[cell*3+c] - g_v_star[cell*3+c]) : T(0.);
    g_out[cell*3+c] = g_Grad[cell*3+c] + mr;
  }
}

// ---------------------------------------------------------------------------
template <typename T>
bool GpuMpmSolver<T>::RunHvpSelfTests(GpuMpmState<T>* s, const T& dt) const {
  // using namespace hvp_detail;
  const int nc = s->num_contacts();
  if (nc == 0) { printf("[hvp] no contacts; skip\n"); return true; }
  const uint32_t tc = s->grid_touched_cnt_host() * s->grid_config().G_BLOCK_VOLUME;
  // per-CELL count (G_DOMAIN_VOLUME), NOT G_GRID_VOLUME (= block count)
  const size_t NG = (size_t(1) << (s->grid_config().DOMAIN_BITS * 3));
  
  T *p,*q,*Hp,*Hq,*gp,*gm,*vbase;
  auto A=[&](T**x){ CUDA_SAFE_CALL(cudaMalloc(x, sizeof(T)*3*NG));
                    CUDA_SAFE_CALL(cudaMemset(*x,0,sizeof(T)*3*NG)); };
  A(&p);A(&q);A(&Hp);A(&Hq);A(&gp);A(&gm);A(&vbase);

  fill_random_touched(s, p);   // host helper: random on touched cells, 0 elsewhere
  fill_random_touched(s, q);

  this->PrecomputeContactHessian(s, dt);

  auto dot=[&](const T*a,const T*b)->T{
    T *d,h=0; CUDA_SAFE_CALL(cudaMalloc(&d,sizeof(T))); CUDA_SAFE_CALL(cudaMemset(d,0,sizeof(T)));
    if(tc>0) dot_touched_kernel<T><<<grid_for(tc),kBlk>>>(s->grid_config(),tc,s->grid_touched_ids(),a,b,d);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(&h,d,sizeof(T),cudaMemcpyDeviceToHost)); CUDA_SAFE_CALL(cudaFree(d)); return h; };

  bool ok=true;

  // (1) finite-difference vs the SAME gradient (lagged-friction consistent)
  CUDA_SAFE_CALL(cudaMemcpy(vbase, s->grid_momentum(), sizeof(T)*3*NG, cudaMemcpyDeviceToDevice));
  const T eps=T(1e-6);
  if(tc>0) axpy_touched_kernel<T><<<grid_for(tc),kBlk>>>(s->grid_config(),tc,s->grid_touched_ids(),vbase,p,+eps,s->grid_momentum());
  this->AssembleContactGradient(s, dt, gp);
  if(tc>0) axpy_touched_kernel<T><<<grid_for(tc),kBlk>>>(s->grid_config(),tc,s->grid_touched_ids(),vbase,p,-eps,s->grid_momentum());
  this->AssembleContactGradient(s, dt, gm);
  CUDA_SAFE_CALL(cudaMemcpy(s->grid_momentum(), vbase, sizeof(T)*3*NG, cudaMemcpyDeviceToDevice));
  this->PrecomputeContactHessian(s, dt);   // restore H at the base point
  this->ApplyHessian(s, p, Hp);
  {
    T *rs,*hs,rh=0,hh=0; CUDA_SAFE_CALL(cudaMalloc(&rs,sizeof(T))); CUDA_SAFE_CALL(cudaMalloc(&hs,sizeof(T)));
    CUDA_SAFE_CALL(cudaMemset(rs,0,sizeof(T))); CUDA_SAFE_CALL(cudaMemset(hs,0,sizeof(T)));
    if(tc>0) fd_resid_kernel<T><<<grid_for(tc),kBlk>>>(s->grid_config(),tc,s->grid_touched_ids(),gp,gm,Hp,T(1)/(2*eps),rs,hs);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(&rh,rs,sizeof(T),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&hh,hs,sizeof(T),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(rs)); CUDA_SAFE_CALL(cudaFree(hs));
    T rel = sqrt(rh) / (sqrt(hh) + T(1e-30));
    printf("[hvp] finite-diff rel-err = %.3e  %s\n", double(rel), rel<1e-3?"PASS":"FAIL");
    ok &= (rel<1e-3);
  }

  // (2) symmetry: p^T H q == q^T H p
  this->ApplyHessian(s,p,Hp); this->ApplyHessian(s,q,Hq);
  { T pHq=dot(p,Hq), qHp=dot(q,Hp);
    T rel=fabs(pHq-qHp)/(fabs(pHq)+fabs(qHp)+T(1e-30));
    printf("[hvp] symmetry    rel-err = %.3e  (pHq=%.6e qHp=%.6e) %s\n",
           double(rel),double(pHq),double(qHp), rel<1e-4?"PASS":"FAIL");
    ok &= (rel<1e-4); }

  // (3) PD: p^T H p > 0
  { this->ApplyHessian(s,p,Hp); T pHp=dot(p,Hp);
    printf("[hvp] pos-def     p^T H p = %.6e  %s\n", double(pHp), pHp>0?"PASS":"FAIL");
    ok &= (pHp>0); }

  cudaFree(p);cudaFree(q);cudaFree(Hp);cudaFree(Hq);cudaFree(gp);cudaFree(gm);cudaFree(vbase);
  printf("[hvp] OVERALL: %s\n", ok?"PASS":"FAIL");
  return ok;
}

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake
