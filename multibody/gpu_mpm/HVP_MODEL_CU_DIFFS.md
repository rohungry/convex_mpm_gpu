# Exact insertions into cuda_mpm_model.cu (and the .cuh members/accessors)

All four buffers, with exact anchor lines from your files. Sized using the same
convenience types the existing allocations use (Vec3<T>/Mat3<T>), so byte counts
match d_g_Grad_ / d_contact_pos_ exactly.

================================================================================
A) cuda_mpm_model.cuh  —  add members (private:, by the other grid/contact ptrs)
================================================================================

After:
    T* d_g_v_star_ = nullptr;
add:
    // matrix-free Hessian-vector product (Hv) grid vectors
    T* d_g_P_  = nullptr;   // CG vector p
    T* d_g_Hp_ = nullptr;   // H p

After:
    T* d_contact_rigid_p_WB_ = nullptr;
add:
    T* d_contact_Hess_    = nullptr;  // H_c^W per contact (Mat3)
    T* d_contact_scratch_ = nullptr;  // J p / H_c^W (J p)  (Vec3)

--- accessors (public:, near grid_Dir() / contact_pos()) ---

After:
    T* grid_Dir()  { return d_g_Dir_;  }
add:
    T* grid_P()  { return d_g_P_;  }
    T* grid_Hp() { return d_g_Hp_; }

After:
    uint32_t* contact_sort_ids() { return d_contact_sort_ids_; }
add:
    T* contact_Hess()    { return d_contact_Hess_; }
    T* contact_scratch() { return d_contact_scratch_; }

================================================================================
B) cuda_mpm_model.cu  —  Finalize(): alloc grid vectors
================================================================================

Find this block (verbatim in your file):

    CUDA_SAFE_CALL(cudaMalloc(&d_g_Hess_, grid_config_.G_DOMAIN_VOLUME * sizeof(Mat3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&d_g_Grad_, grid_config_.G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&d_g_Dir_, grid_config_.G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&d_g_v_star_, grid_config_.G_DOMAIN_VOLUME * sizeof(Vec3<T>)));

Insert immediately AFTER it:

    // Hv grid vectors (same per-cell sizing as d_g_Grad_)
    CUDA_SAFE_CALL(cudaMalloc(&d_g_P_,  grid_config_.G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&d_g_Hp_, grid_config_.G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMemset(d_g_P_,  0, grid_config_.G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMemset(d_g_Hp_, 0, grid_config_.G_DOMAIN_VOLUME * sizeof(Vec3<T>)));

================================================================================
C) cuda_mpm_model.cu  —  Destroy(): free grid vectors
================================================================================

Find:

    CUDA_SAFE_CALL(cudaFree(d_g_Hess_));
    CUDA_SAFE_CALL(cudaFree(d_g_Grad_));
    CUDA_SAFE_CALL(cudaFree(d_g_Dir_));
    CUDA_SAFE_CALL(cudaFree(d_g_v_star_));
    d_g_Hess_ = nullptr;
    d_g_Grad_ = nullptr;
    d_g_Dir_ = nullptr;
    d_g_v_star_ = nullptr;

Insert AFTER it:

    CUDA_SAFE_CALL(cudaFree(d_g_P_));
    CUDA_SAFE_CALL(cudaFree(d_g_Hp_));
    d_g_P_  = nullptr;
    d_g_Hp_ = nullptr;

================================================================================
D) cuda_mpm_model.cu  —  Destroy(): free contact buffers
================================================================================

Find the contact free block; after this (note your file's existing copy/paste
bug: the d_contact_sort_ids_ branch sets d_contact_sort_keys_ = nullptr — leave
that as-is or fix separately, not our concern):

    if (d_contact_sort_ids_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_sort_ids_));
        d_contact_sort_keys_ = nullptr;
    }

Insert AFTER it:

    if (d_contact_Hess_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_Hess_));
        d_contact_Hess_ = nullptr;
    }
    if (d_contact_scratch_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_scratch_));
        d_contact_scratch_ = nullptr;
    }

================================================================================
E) cuda_mpm_model.cu  —  ReallocateContacts(): free + alloc
================================================================================

E1. In the "grow" branch, the free section. After:

        if (d_contact_sort_ids_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_sort_ids_));
        }

insert:

        if (d_contact_Hess_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_Hess_));
        }
        if (d_contact_scratch_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_scratch_));
        }

E2. The alloc section (note: this block uses bare cudaMalloc, matching the file).
After:

        cudaMalloc(&d_contact_sort_ids_, sizeof(uint32_t) * contact_buffer_size);

insert:

        cudaMalloc(&d_contact_Hess_,    sizeof(Mat3<T>) * contact_buffer_size);
        cudaMalloc(&d_contact_scratch_, sizeof(Vec3<T>) * contact_buffer_size);

================================================================================
F) cuda_mpm_model.cu  —  fill_random_touched_impl + explicit instantiation
================================================================================

The Hv .inc.cu (included into cuda_mpm_solver.cu) declares
`fill_random_touched_impl<T>` and calls it. Define it ONCE. Cleanest spot: put
the definition in cuda_mpm_model.cu (it's state-buffer code), just ABOVE the
existing line:

    template class GpuMpmState<config::GpuT>;

Add:

    template <typename T>
    void fill_random_touched_impl(GpuMpmState<T>* s, T* dst_dev) {
        const auto& gc = s->grid_config();
        const uint32_t n_blocks = s->grid_touched_cnt_host();
        const size_t n_cells = size_t(1) << (gc.DOMAIN_BITS * 3);  // G_DOMAIN_VOLUME
        std::vector<uint32_t> h_ids(n_blocks);
        CUDA_SAFE_CALL(cudaMemcpy(h_ids.data(), s->grid_touched_ids(),
                                  sizeof(uint32_t) * n_blocks, cudaMemcpyDeviceToHost));
        std::vector<T> h(3 * n_cells, T(0));
        std::mt19937 rng(12345u);
        std::normal_distribution<double> Nd(0.0, 1.0);
        const uint32_t bvol = gc.G_BLOCK_VOLUME;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t block_idx = h_ids[b];
            for (uint32_t local = 0; local < bvol; ++local) {
                const uint32_t cell = (block_idx << (gc.G_BLOCK_BITS * 3)) | local;
                h[cell*3+0] = T(Nd(rng));
                h[cell*3+1] = T(Nd(rng));
                h[cell*3+2] = T(Nd(rng));
            }
        }
        CUDA_SAFE_CALL(cudaMemcpy(dst_dev, h.data(), sizeof(T) * 3 * n_cells,
                                  cudaMemcpyHostToDevice));
    }
    template void fill_random_touched_impl<config::GpuT>(GpuMpmState<config::GpuT>*, config::GpuT*);

Add at the top of cuda_mpm_model.cu with the other includes:

    #include <random>
    #include <vector>

NOTE on the macro: the .inc.cu has `#define fill_random_touched(s,dst)
fill_random_touched_impl((s),(dst))` and calls fill_random_touched(s, p). Since
cuda_mpm_model.cu defines the *_impl directly (not via the macro), there is no
conflict — the macro lives only in the solver TU.

================================================================================
G) Sanity: G_DOMAIN_VOLUME vs G_GRID_VOLUME (the bug already avoided)
================================================================================

Confirmed from your Finalize(): per-cell grids (masses, momentum, Hess, Grad,
Dir, v_star) are all G_DOMAIN_VOLUME; only touched_flags / touched_ids are
G_GRID_VOLUME (block count). The Hv grid vectors are per-cell => G_DOMAIN_VOLUME.
The self-test scratch in cuda_mpm_hvp.inc.cu already uses
(1 << (DOMAIN_BITS*3)) = G_DOMAIN_VOLUME. Consistent.
