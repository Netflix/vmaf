#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "feature/common/macros.h"
#include "feature/integer_vif.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static FORCE_INLINE inline void
pad_top_and_bottom(VifBuffer buf, unsigned h, int fwidth)
{
    const unsigned fwidth_half = fwidth / 2;
    unsigned char *ref = buf.ref;
    unsigned char *dis = buf.dis;
    for (unsigned i = 1; i <= fwidth_half; ++i) {
        size_t offset = buf.stride * i;
        memcpy(ref - offset, ref + offset, buf.stride);
        memcpy(dis - offset, dis + offset, buf.stride);
        memcpy(ref + buf.stride * (h - 1) + buf.stride * i,
               ref + buf.stride * (h - 1) - buf.stride * i,
               buf.stride);
        memcpy(dis + buf.stride * (h - 1) + buf.stride * i,
               dis + buf.stride * (h - 1) - buf.stride * i,
               buf.stride);
    }
}

static FORCE_INLINE inline void
decimate_and_pad(VifBuffer buf, unsigned w, unsigned h, int scale)
{
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t mu_stride = buf.stride_16 / sizeof(uint16_t);

    for (unsigned i = 0; i < h / 2; ++i) {
        for (unsigned j = 0; j < w / 2; ++j) {
            ref[i * stride + j] = buf.mu1[(i * 2) * mu_stride + (j * 2)];
            dis[i * stride + j] = buf.mu2[(i * 2) * mu_stride + (j * 2)];
        }
    }
    pad_top_and_bottom(buf, h / 2, vif_filter1d_width[scale]);
}


// 32 BITS macros //

// The macro instances and loads 4 adjacent uint32x4_t vectors
#define NEON_FILTER_INSTANCE_AND_LOAD_U32X4_LU4(vec_name, pointer) \
    uint32x4_t vec_name##_0 = vld1q_u32(pointer);                  \
    uint32x4_t vec_name##_1 = vld1q_u32((pointer + 4));            \
    uint32x4_t vec_name##_2 = vld1q_u32((pointer + 8));            \
    uint32x4_t vec_name##_3 = vld1q_u32((pointer + 12));

// The macro instances 4 accumulators as multplication of 4 adjacent vectors by scalar
#define NEON_FILTER_INSTANCE_U32X4_MULL_U32X4_WITH_CONST_LU4(vec_accum_name, vec_init, vec_name, const_val) \
    uint32x4_t vec_accum_name##_0 = vmlaq_n_u32(vec_init, vec_name##_0, const_val);                         \
    uint32x4_t vec_accum_name##_1 = vmlaq_n_u32(vec_init, vec_name##_1, const_val);                         \
    uint32x4_t vec_accum_name##_2 = vmlaq_n_u32(vec_init, vec_name##_2, const_val);                         \
    uint32x4_t vec_accum_name##_3 = vmlaq_n_u32(vec_init, vec_name##_3, const_val);

// The macro loads 4 adjacent uint32x4_t vectors
#define NEON_FILTER_LOAD_U32X4_LU4(vec_name, pointer) \
    vec_name##_0 = vld1q_u32(pointer);                \
    vec_name##_1 = vld1q_u32((pointer + 4));          \
    vec_name##_2 = vld1q_u32((pointer + 8));          \
    vec_name##_3 = vld1q_u32((pointer + 12));

// The macro updates 4 accumulators with multplications of 4 adjacent vectors by scalar
#define NEON_FILTER_UPDATE_U32X4_ACCUM_MULL_U32X4_WITH_CONST_LU4(vec_accum_name, vec_name, const_val) \
    vec_accum_name##_0 = vmlaq_n_u32(vec_accum_name##_0, vec_name##_0, const_val);                    \
    vec_accum_name##_1 = vmlaq_n_u32(vec_accum_name##_1, vec_name##_1, const_val);                    \
    vec_accum_name##_2 = vmlaq_n_u32(vec_accum_name##_2, vec_name##_2, const_val);                    \
    vec_accum_name##_3 = vmlaq_n_u32(vec_accum_name##_3, vec_name##_3, const_val);

// The macro takes an accumulator of uint32x4_t, adds an offset, shift it right and stores the result.
#define NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_name, offset_vec, shift_vec, store_pointer) \
    accum_name = vaddq_u32(accum_name, offset_vec);                                            \
    accum_name = vshlq_u32(accum_name, shift_vec);                                             \
    vst1q_u32(store_pointer, accum_name);

// The macro uses the above macro and calls it 4 times - for loop unrool (LU2) and low and high (HI_LO)
#define NEON_FILTER_OFFSET_SHIFT_STORE_U32X4_HI_LO_LU2(accum_name, offset_vec, shift_vec, store_pointer) \
    NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_name##_0_l, offset_vec, shift_vec, store_pointer)         \
    NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_name##_0_h, offset_vec, shift_vec, (store_pointer + 4))   \
    NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_name##_1_l, offset_vec, shift_vec, (store_pointer + 8))   \
    NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_name##_1_h, offset_vec, shift_vec, (store_pointer + 12))

// The macro takes an accumulator of uint32x4_t, shift it right and stores the result.
#define NEON_FILTER_SHIFT_STORE_U32X4(accum_name, shift_vec, store_pointer) \
    accum_name = vshlq_u32(accum_name, shift_vec);                          \
    vst1q_u32(store_pointer, accum_name);

// The macro uses the above macro and calls it 4 times - for loop unrool (LU2) and low and high (HI_LO)
#define NEON_FILTER_SHIFT_STORE_U32X4_HI_LO_LU2(accum_name, shift_vec, store_pointer) \
    NEON_FILTER_SHIFT_STORE_U32X4(accum_name##_0_l, shift_vec, store_pointer)         \
    NEON_FILTER_SHIFT_STORE_U32X4(accum_name##_0_h, shift_vec, (store_pointer + 4))   \
    NEON_FILTER_SHIFT_STORE_U32X4(accum_name##_1_l, shift_vec, (store_pointer + 8))   \
    NEON_FILTER_SHIFT_STORE_U32X4(accum_name##_1_h, shift_vec, (store_pointer + 12))

// The macro uses the above macro and calls it 4 times - for low high (LH) and low and high (HI_LO)
#define NEON_FILTER_SHIFT_STORE_U32X4_HI_LO_LH(accum_name, shift_vec, store_pointer) \
    NEON_FILTER_SHIFT_STORE_U32X4(accum_name##_l_l, shift_vec, store_pointer)        \
    NEON_FILTER_SHIFT_STORE_U32X4(accum_name##_l_h, shift_vec, (store_pointer + 4))  \
    NEON_FILTER_SHIFT_STORE_U32X4(accum_name##_h_l, shift_vec, (store_pointer + 8))  \
    NEON_FILTER_SHIFT_STORE_U32X4(accum_name##_h_h, shift_vec, (store_pointer + 12))

// The macro instance and initialize 2 accumulators (low and high) with the multiplication of a uint16x8_t vector by a scalar (filter element)
#define NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_name, vec_name, const_val) \
    uint32x4_t accum_name##_l = vmull_n_u16(vget_low_u16(vec_name), const_val);                         \
    uint32x4_t accum_name##_h = vmull_high_n_u16(vec_name, const_val);

// Using the macro above with loop unrool (LU2)
#define NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(accum_name, vec_name, const_val) \
    NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_name##_0, vec_name##_0, const_val) \
    NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_name##_1, vec_name##_1, const_val)

// The macro instance and initialize 2 accumulators (low and high) with the multiplication of a uint16x8_t vector by a scalar (filter element)
#define NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_name, vec_init, vec_name, const_val) \
    uint32x4_t accum_name##_l = vmlal_n_u16(vec_init, vget_low_u16(vec_name), const_val);                      \
    uint32x4_t accum_name##_h = vmlal_high_n_u16(vec_init, vec_name, const_val);

// Using the macro above with loop unrool (LU2)
#define NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(accum_name, vec_init, vec_name, const_val) \
    NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_name##_0, vec_init, vec_name##_0, const_val) \
    NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_name##_1, vec_init, vec_name##_1, const_val)

// Using the macro above with low and high vectors (LH)
#define NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI_LH(accum_name, vec_init, vec_name, const_val)  \
    NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_name##_l, vec_init, vec_name##_l, const_val) \
    NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_name##_h, vec_init, vec_name##_h, const_val)

// The macro updates 2 accumulators (low and high) with the original accumlator values and adds the multiplication of a uint16x8_t vector by a scalar (filter element)
#define NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI(accum_name, vec_name, const_val) \
    accum_name##_l = vmlal_n_u16(accum_name##_l, vget_low_u16(vec_name), const_val);     \
    accum_name##_h = vmlal_high_n_u16(accum_name##_h, vec_name, const_val);

// Using the macro above with loop unrool (LU2)
#define NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LU2(accum_name, vec_name, const_val) \
    NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI(accum_name##_0, vec_name##_0, const_val) \
    NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI(accum_name##_1, vec_name##_1, const_val)

// Using the macro above with low and high vectors (LH)
#define NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LH(accum_name, vec_name, const_val)  \
    NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI(accum_name##_l, vec_name##_l, const_val) \
    NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI(accum_name##_h, vec_name##_h, const_val)

// The macro takes 4 adjacent accumulators, adds offset, shift them, unzip them (pair-wise) into 2 uint16x8_t vectors, and stores them
#define NEON_FILTER_SHIFT_UNZIP_STORE_U32X4_TO_U16X8_LU2(vec_name, shift_vec, store_address)                 \
    {                                                                                                        \
        uint16x8_t vec_name##_tmp_0 = vuzp1q_u16(vreinterpretq_u16_u32(vshlq_u32(vec_name##_0, shift_vec)),  \
                                                 vreinterpretq_u16_u32(vshlq_u32(vec_name##_1, shift_vec))); \
        uint16x8_t vec_name##_tmp_1 = vuzp1q_u16(vreinterpretq_u16_u32(vshlq_u32(vec_name##_2, shift_vec)),  \
                                                 vreinterpretq_u16_u32(vshlq_u32(vec_name##_3, shift_vec))); \
        vst1q_u16((store_address), vec_name##_tmp_0);                                                        \
        vst1q_u16((store_address + 8), vec_name##_tmp_1);                                                    \
    }

// 64 BITS macros //

// The macro instance and initialize 2 accumulators (low and high) with the multiplication of 2 uint32x4_t vectors
#define NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_LO_HI(accum_name, init_vec, vec_name_a, vec_name_b)   \
    uint64x2_t accum_name##_l = vmlal_u32(init_vec, vget_low_u32(vec_name_a), vget_low_u32(vec_name_b)); \
    uint64x2_t accum_name##_h = vmlal_high_u32(init_vec, vec_name_a, vec_name_b);

// The macro instance and initialize 2 accumulators (low and high) with the multiplication of a uint32x4_t vector by a scalar (filter element)
#define NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI(accum_name, init_vec, vec_name, const_val) \
    uint64x2_t accum_name##_l = vmlal_n_u32(init_vec, vget_low_u32(vec_name), const_val);                      \
    uint64x2_t accum_name##_h = vmlal_high_n_u32(init_vec, vec_name, const_val);

// Using the macro above with loop unrool (LU2)
#define NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LU2(accum_name, init_vec, vec_name, const_val) \
    NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI(accum_name##_0, init_vec, vec_name##_0, const_val) \
    NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI(accum_name##_1, init_vec, vec_name##_1, const_val)

// Using the macro above with low and high vectors (LH)
#define NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LH(accum_name, init_vec, vec_name, const_val)  \
    NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI(accum_name##_l, init_vec, vec_name##_l, const_val) \
    NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI(accum_name##_h, init_vec, vec_name##_h, const_val)

// The macro updates 2 accumulators (low and high) with the original accumlator values and adds the multiplication of 2 uint32x4_t vectors
#define NEON_FILTER_UPDATE_U64X2_ACCUM_LO_HI(accum_name, vec_name_a, vec_name_b)                    \
    accum_name##_l = vmlal_u32(accum_name##_l, vget_low_u32(vec_name_a), vget_low_u32(vec_name_b)); \
    accum_name##_h = vmlal_high_u32(accum_name##_h, vec_name_a, vec_name_b);

// The macro updates 2 accumulators (low and high) with the original accumlator values and adds the multiplication of a uint32x4_t vector by a scalar (filter element)
#define NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI(accum_name, vec_name, const_val) \
    accum_name##_l = vmlal_n_u32(accum_name##_l, vget_low_u32(vec_name), const_val);     \
    accum_name##_h = vmlal_high_n_u32(accum_name##_h, vec_name, const_val);

// Using the macro above with loop unrool (LU2)
#define NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI_LU2(accum_name, vec_name, const_val) \
    NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI(accum_name##_0, vec_name##_0, const_val) \
    NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI(accum_name##_1, vec_name##_1, const_val)

// Using the macro above with low and high vectors (LH)
#define NEON_FILTER_ACCUM_LO_HI_LH_U64X2_WITH_CONST_LH(accum_name, vec_name, const_val)      \
    NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI(accum_name##_l, vec_name##_l, const_val) \
    NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI(accum_name##_h, vec_name##_h, const_val)

// The macro mul 2 low and high 32-bits accumulators, shift them, unzip them into single uint32x4_t vector
#define NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(out_vec_name, init_vec, vec_name_in_a, vec_name_in_b, shift_vec)   \
    uint64x2_t out_vec_name##_l = vmlal_u32(init_vec, vget_low_u32(vec_name_in_a), vget_low_u32(vec_name_in_b)); \
    uint64x2_t out_vec_name##_h = vmlal_high_u32(init_vec, vec_name_in_a, vec_name_in_b); \
    uint32x4_t out_vec_name = vuzp1q_u32(vreinterpretq_u32_u64(vshlq_u64(out_vec_name##_l, shift_vec)),  \
                                             vreinterpretq_u32_u64(vshlq_u64(out_vec_name##_h, shift_vec))); 

// The macro takes low and high accumulators, shift them, unzip them into single uint32x4_t vector
#define NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_name, shift_vec, out_name)           \
    uint32x4_t out_name = vuzp1q_u32(vreinterpretq_u32_u64(vshlq_u64(accum_name##_l, shift_vec)),  \
                                             vreinterpretq_u32_u64(vshlq_u64(accum_name##_h, shift_vec))); 


// The macro takes low and high accumulators, shift them, unzip them into single uint32x4_t vector, and stores it
#define NEON_FILTER_SHIFT_UNZIP_STORE_U64X2_TO_U32X4_LO_HI(accum_name, shift_vec, store_pointer)           \
    uint32x4_t accum_name##_tmp = vuzp1q_u32(vreinterpretq_u32_u64(vshlq_u64(accum_name##_l, shift_vec)),  \
                                             vreinterpretq_u32_u64(vshlq_u64(accum_name##_h, shift_vec))); \
    vst1q_u32(store_pointer, accum_name##_tmp);

// Miscellaneous //

// The macro loads 2 adajcent uint8x8_t vectors, moves them to 2 uint16x8_t vectors, and multplies each by itself into sqr vec
#define NEON_FILTER_LOAD_U8X8_MOVE_TO_U16X8_AND_SQR_LU2(vec_name, vec16_name, sqr_vec_name, load_address) \
    uint8x8_t vec_name##_0 = vld1_u8(load_address);                                                       \
    uint8x8_t vec_name##_1 = vld1_u8((load_address + 8));                                                 \
    uint16x8_t vec16_name##_0 = vmovl_u8(vec_name##_0);                                                   \
    uint16x8_t vec16_name##_1 = vmovl_u8(vec_name##_1);                                                   \
    uint16x8_t sqr_vec_name##_0 = vmull_u8(vec_name##_0, vec_name##_0);                                   \
    uint16x8_t sqr_vec_name##_1 = vmull_u8(vec_name##_1, vec_name##_1);

// The macro loads uint16x8_t vector, moves it to 2 uint32x4_t vectors, and multplies each by itself into sqr vec
#define NEON_FILTER_LOAD_U16X8_AND_MOVE_TO_U32X4_AND_SQR(vec_name, vec32_name, sqr_vec_name, load_address) \
    uint16x8_t vec_name = vld1q_u16(load_address);                                                         \
    uint32x4_t vec32_name##_l = vmovl_u16(vget_low_u16(vec_name));                                         \
    uint32x4_t vec32_name##_h = vmovl_high_u16(vec_name);                                                  \
    uint32x4_t sqr_vec_name##_l = vmulq_u32(vec32_name##_l, vec32_name##_l);                               \
    uint32x4_t sqr_vec_name##_h = vmulq_u32(vec32_name##_h, vec32_name##_h);

// End of macros

void vif_subsample_rd_8_neon(VifBuffer buf, unsigned int w, unsigned int h)
{
    const unsigned int uiw15 = (w > 15 ? w - 15 : 0);
    const unsigned int fwidth = vif_filter1d_width[1];
    const uint16_t *vif_filt_s1 = vif_filter1d_table[1];
    const uint8_t *ref = (uint8_t *)buf.ref;
    const uint8_t *dis = (uint8_t *)buf.dis;
    const ptrdiff_t dst_stride = buf.stride_16 / sizeof(uint16_t);
    ptrdiff_t i_dst_stride = 0;

    const uint32x4_t offset_vec_v = vdupq_n_u32(128);
    const int32x4_t shift_vec_v = vdupq_n_s32(-8);
    const uint32x4_t offset_vec_h = vdupq_n_u32(32768);
    const int32x4_t shift_vec_h = vdupq_n_s32(-16);

    for (unsigned int i = 0; i < h; ++i, i_dst_stride += dst_stride)
    {

        int ii = i - fwidth / 2;
        const uint8_t *p_ref = ref + ii * buf.stride;
        const uint8_t *p_dis = dis + ii * buf.stride;

        // VERTICAL Neon
        unsigned int j = 0;
        for (; j < uiw15; j += 16, p_ref += 16, p_dis += 16)
        {
            uint8x8_t ref_vec_8u_0 = vld1_u8(p_ref);
            uint16x8_t ref_vec_16u_0 = vmovl_u8(ref_vec_8u_0);
            uint8x8_t dis_vec_8u_0 = vld1_u8(p_dis);
            uint16x8_t dis_vec_16u_0 = vmovl_u8(dis_vec_8u_0);
            uint8x8_t ref_vec_8u_1 = vld1_u8(p_ref + 8);
            uint16x8_t ref_vec_16u_1 = vmovl_u8(ref_vec_8u_1);
            uint8x8_t dis_vec_8u_1 = vld1_u8(p_dis + 8);
            uint16x8_t dis_vec_16u_1 = vmovl_u8(dis_vec_8u_1);

            NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(accum_f_ref, offset_vec_v, ref_vec_16u, vif_filt_s1[0]);
            NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(accum_f_dis, offset_vec_v, dis_vec_16u, vif_filt_s1[0]);

            const uint8_t *pp_ref = p_ref + buf.stride;
            const uint8_t *pp_dis = p_dis + buf.stride;
            for (unsigned fi = 1; fi < fwidth; ++fi, pp_ref += buf.stride, pp_dis += buf.stride)
            {
                uint8x8_t ref_vec_8u_0 = vld1_u8(pp_ref);
                uint16x8_t ref_vec_16u_0 = vmovl_u8(ref_vec_8u_0);
                uint8x8_t ref_vec_8u_1 = vld1_u8(pp_ref + 8);
                uint16x8_t ref_vec_16u_1 = vmovl_u8(ref_vec_8u_1);

                uint8x8_t dis_vec_8u_0 = vld1_u8(pp_dis);
                uint16x8_t dis_vec_16u_0 = vmovl_u8(dis_vec_8u_0);
                uint8x8_t dis_vec_8u_1 = vld1_u8(pp_dis + 8);
                uint16x8_t dis_vec_16u_1 = vmovl_u8(dis_vec_8u_1);

                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LU2(accum_f_ref, ref_vec_16u, vif_filt_s1[fi]);
                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LU2(accum_f_dis, dis_vec_16u, vif_filt_s1[fi]);
            }

            NEON_FILTER_SHIFT_STORE_U32X4_HI_LO_LU2(accum_f_ref, shift_vec_v, buf.tmp.ref_convol + j);
            NEON_FILTER_SHIFT_STORE_U32X4_HI_LO_LU2(accum_f_dis, shift_vec_v, buf.tmp.dis_convol + j);
        }

        // Scalar code for Vertical leftover.
        for (; j < w; ++j)
        {
            uint32_t accum_ref = 128;
            uint32_t accum_dis = 128;
            for (unsigned fi = 0; fi < fwidth; ++fi)
            {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s1[fi];
                const uint8_t *ref = (uint8_t *)buf.ref;
                const uint8_t *dis = (uint8_t *)buf.dis;
                accum_ref += fcoeff * (uint32_t)ref[ii_check * buf.stride + j];
                accum_dis += fcoeff * (uint32_t)dis[ii_check * buf.stride + j];
            }
            buf.tmp.ref_convol[j] = accum_ref >> 8;
            buf.tmp.dis_convol[j] = accum_dis >> 8;
        }

        PADDING_SQ_DATA_2(buf, w, fwidth / 2);

        // HORIZONTAL
        uint32_t *pRefConv = (uint32_t *)buf.tmp.ref_convol - (fwidth / 2);
        uint32_t *pDisConv = (uint32_t *)buf.tmp.dis_convol - (fwidth / 2);

        j = 0;
        for (; j < uiw15; j += 16, pRefConv += 16, pDisConv += 16)
        {
            NEON_FILTER_INSTANCE_AND_LOAD_U32X4_LU4(ref_conv_vec_u32, pRefConv);
            NEON_FILTER_INSTANCE_AND_LOAD_U32X4_LU4(dis_conv_vec_u32, pDisConv);

            NEON_FILTER_INSTANCE_U32X4_MULL_U32X4_WITH_CONST_LU4(accum_ref_conv, offset_vec_h, ref_conv_vec_u32, vif_filt_s1[0]);
            NEON_FILTER_INSTANCE_U32X4_MULL_U32X4_WITH_CONST_LU4(accum_dis_conv, offset_vec_h, dis_conv_vec_u32, vif_filt_s1[0]);

            for (unsigned fj = 1; fj < fwidth; ++fj)
            {
                NEON_FILTER_LOAD_U32X4_LU4(ref_conv_vec_u32, pRefConv + fj);
                NEON_FILTER_LOAD_U32X4_LU4(dis_conv_vec_u32, pDisConv + fj);

                NEON_FILTER_UPDATE_U32X4_ACCUM_MULL_U32X4_WITH_CONST_LU4(accum_ref_conv, ref_conv_vec_u32, vif_filt_s1[fj]);
                NEON_FILTER_UPDATE_U32X4_ACCUM_MULL_U32X4_WITH_CONST_LU4(accum_dis_conv, dis_conv_vec_u32, vif_filt_s1[fj]);
            }

            NEON_FILTER_SHIFT_UNZIP_STORE_U32X4_TO_U16X8_LU2(accum_ref_conv, shift_vec_h, buf.mu1 + i_dst_stride + j);
            NEON_FILTER_SHIFT_UNZIP_STORE_U32X4_TO_U16X8_LU2(accum_dis_conv, shift_vec_h, buf.mu2 + i_dst_stride + j)
        }

        // Scalar code for Horizontal leftover.
        for (; j < w; ++j)
        {
            uint32_t accum_ref = 32768;
            uint32_t accum_dis = 32768;
            for (unsigned fj = 0; fj < fwidth; ++fj)
            {
                int jj = j - fwidth / 2;
                int jj_check = jj + fj;
                const uint16_t fcoeff = vif_filt_s1[fj];
                accum_ref += fcoeff * buf.tmp.ref_convol[jj_check];
                accum_dis += fcoeff * buf.tmp.dis_convol[jj_check];
            }
            buf.mu1[i_dst_stride + j] = (uint16_t)(accum_ref >> 16);
            buf.mu2[i_dst_stride + j] = (uint16_t)(accum_dis >> 16);
        }
    }

    decimate_and_pad(buf, w, h, 0);
}



void vif_subsample_rd_16_neon(VifBuffer buf, unsigned int w, unsigned int h, int scale, int bpc)
{
    const unsigned int uiw15 = (w > 15 ? w - 15 : 0);
    const unsigned int fwidth = vif_filter1d_width[scale + 1];
    const uint16_t *vif_filt_s = vif_filter1d_table[scale + 1];
    int32_t add_shift_round_VP, shift_VP;

    if (scale == 0)
    {
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP = bpc;
    }
    else
    {
        add_shift_round_VP = 32768;
        shift_VP = 16;
    }

    const uint32x4_t add_shift_round_VP_vec = vdupq_n_u32(add_shift_round_VP);
    const int32x4_t shift_VP_vec = vdupq_n_s32(-shift_VP);
    const uint32x4_t offset_vec_h = vdupq_n_u32(32768);
    int32x4_t shift_vec_h = vdupq_n_s32(-16);

    const uint16_t *ref = (uint16_t *)buf.ref;
    const uint16_t *dis = (uint16_t *)buf.dis;

    const ptrdiff_t stride_v = buf.stride / sizeof(uint16_t);
    const ptrdiff_t stride_h = buf.stride_16 / sizeof(uint16_t);
    ptrdiff_t i_dst_stride = 0;

    for (unsigned i = 0; i < h; ++i, i_dst_stride += stride_h)
    {

        int ii = i - fwidth / 2;
        const uint16_t *p_ref = ref + ii * stride_v;
        const uint16_t *p_dis = dis + ii * stride_v;

        // VERTICAL Neon
        unsigned int j = 0;
        for (; j < uiw15; j += 16, p_ref += 16, p_dis += 16)
        {
            uint16x8_t ref_vec_16u_l = vld1q_u16(p_ref);
            uint16x8_t ref_vec_16u_h = vld1q_u16(p_ref + 8);
            uint16x8_t dis_vec_16u_l = vld1q_u16(p_dis);
            uint16x8_t dis_vec_16u_h = vld1q_u16(p_dis + 8);

            NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI_LH(accum_f_ref, add_shift_round_VP_vec, ref_vec_16u, vif_filt_s[0]);
            NEON_FILTER_INSTANCE_U32X4_INIT_MULL_U16X4_WITH_CONST_LO_HI_LH(accum_f_dis, add_shift_round_VP_vec, dis_vec_16u, vif_filt_s[0]);

            const uint16_t *pp_ref = p_ref + stride_v;
            const uint16_t *pp_dis = p_dis + stride_v;
            for (unsigned fi = 1; fi < fwidth; ++fi, pp_ref += stride_v, pp_dis += stride_v)
            {
                ref_vec_16u_l = vld1q_u16(pp_ref);
                ref_vec_16u_h = vld1q_u16(pp_ref + 8);
                dis_vec_16u_l = vld1q_u16(pp_dis);
                dis_vec_16u_h = vld1q_u16(pp_dis + 8);

                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LH(accum_f_ref, ref_vec_16u, vif_filt_s[fi]);
                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LH(accum_f_dis, dis_vec_16u, vif_filt_s[fi]);
            }

            NEON_FILTER_SHIFT_STORE_U32X4_HI_LO_LH(accum_f_ref, shift_VP_vec, buf.tmp.ref_convol + j);
            NEON_FILTER_SHIFT_STORE_U32X4_HI_LO_LH(accum_f_dis, shift_VP_vec, buf.tmp.dis_convol + j);
        }

        // Scalar code for Vertical leftover.
        for (; j < w; ++j)
        {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi)
            {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s[fi];
                uint16_t *ref = (uint16_t *)buf.ref;
                uint16_t *dis = (uint16_t *)buf.dis;
                accum_ref += fcoeff * ((uint32_t)ref[ii_check * stride_v + j]);
                accum_dis += fcoeff * ((uint32_t)dis[ii_check * stride_v + j]);
            }
            buf.tmp.ref_convol[j] = (uint16_t)((accum_ref + add_shift_round_VP) >> shift_VP);
            buf.tmp.dis_convol[j] = (uint16_t)((accum_dis + add_shift_round_VP) >> shift_VP);
        }

        PADDING_SQ_DATA_2(buf, w, fwidth / 2);

        // HORIZONTAL
        uint32_t *pRefConv = (uint32_t *)buf.tmp.ref_convol - (fwidth / 2);
        uint32_t *pDisConv = (uint32_t *)buf.tmp.dis_convol - (fwidth / 2);

        j = 0;
        for (; j < uiw15; j += 16, pRefConv += 16, pDisConv += 16)
        {
            NEON_FILTER_INSTANCE_AND_LOAD_U32X4_LU4(ref_conv_vec_u32, pRefConv);
            NEON_FILTER_INSTANCE_AND_LOAD_U32X4_LU4(dis_conv_vec_u32, pDisConv);

            NEON_FILTER_INSTANCE_U32X4_MULL_U32X4_WITH_CONST_LU4(accum_ref_conv, offset_vec_h, ref_conv_vec_u32, vif_filt_s[0]);
            NEON_FILTER_INSTANCE_U32X4_MULL_U32X4_WITH_CONST_LU4(accum_dis_conv, offset_vec_h, dis_conv_vec_u32, vif_filt_s[0]);

            for (unsigned fj = 1; fj < fwidth; ++fj)
            {
                NEON_FILTER_LOAD_U32X4_LU4(ref_conv_vec_u32, pRefConv + fj);
                NEON_FILTER_LOAD_U32X4_LU4(dis_conv_vec_u32, pDisConv + fj);

                NEON_FILTER_UPDATE_U32X4_ACCUM_MULL_U32X4_WITH_CONST_LU4(accum_ref_conv, ref_conv_vec_u32, vif_filt_s[fj]);
                NEON_FILTER_UPDATE_U32X4_ACCUM_MULL_U32X4_WITH_CONST_LU4(accum_dis_conv, dis_conv_vec_u32, vif_filt_s[fj]);
            }

            NEON_FILTER_SHIFT_UNZIP_STORE_U32X4_TO_U16X8_LU2(accum_ref_conv, shift_vec_h, buf.mu1 + i_dst_stride + j);
            NEON_FILTER_SHIFT_UNZIP_STORE_U32X4_TO_U16X8_LU2(accum_dis_conv, shift_vec_h, buf.mu2 + i_dst_stride + j);
        }

        // Scalar code for Horizontal leftover.
        for (; j < w; ++j)
        {
            uint32_t accum_ref = 32768;
            uint32_t accum_dis = 32768;
            for (unsigned fj = 0; fj < fwidth; ++fj)
            {
                int jj = j - fwidth / 2;
                int jj_check = jj + fj;
                const uint16_t fcoeff = vif_filt_s[fj];
                accum_ref += fcoeff * buf.tmp.ref_convol[jj_check];
                accum_dis += fcoeff * buf.tmp.dis_convol[jj_check];
            }
            buf.mu1[i_dst_stride + j] = (uint16_t)(accum_ref >> 16);
            buf.mu2[i_dst_stride + j] = (uint16_t)(accum_dis >> 16);
        }
    }

    decimate_and_pad(buf, w, h, scale);
}


void vif_statistic_8_neon(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h)
{
    const unsigned int uiw15 = (w > 15 ? w - 15 : 0);
    const unsigned int uiw7 = (w > 7 ? w - 7 : 0);
    const unsigned int fwidth = vif_filter1d_width[0];
    const uint16_t *vif_filt_s0 = vif_filter1d_table[0];
    VifBuffer buf = s->buf;
    uint16_t *log2_table = s->log2_table;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;

    int64_t accum_num_log = 0.0;
    int64_t accum_den_log = 0.0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;
    static const int32_t sigma_nsq = 65536 << 1;

    const uint8_t *ref = (uint8_t *)buf.ref;
    const uint8_t *dis = (uint8_t *)buf.dis;
    const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
    ptrdiff_t i_dst_stride = 0;

    const uint32x4_t offset_vec_v = vdupq_n_u32(128);
    const int32x4_t shift_vec_v = vdupq_n_s32(-8);

    const uint64x2_t offset_vec_h = vdupq_n_u64(32768);
    const int64x2_t shift_vec_h = vdupq_n_s64(-16);

    int32_t xx[8], yy[8], xy[8];

    for (unsigned i = 0; i < h; ++i, i_dst_stride += dst_stride)
    {
        int ii = i - fwidth / 2;
        const uint8_t *p_ref = ref + ii * buf.stride;
        const uint8_t *p_dis = dis + ii * buf.stride;

        // VERTICAL Neon
        unsigned int j = 0;
        for (; j < uiw15; j += 16, p_ref += 16, p_dis += 16)
        {
            NEON_FILTER_LOAD_U8X8_MOVE_TO_U16X8_AND_SQR_LU2(ref_vec_8u, ref_vec_16u, ref_ref_vec_16u, p_ref);
            NEON_FILTER_LOAD_U8X8_MOVE_TO_U16X8_AND_SQR_LU2(dis_vec_8u, dis_vec_16u, dis_dis_vec_16u, p_dis);

            NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(accum_f_ref, ref_vec_16u, vif_filt_s0[0])
            NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(accum_f_dis, dis_vec_16u, vif_filt_s0[0]);
            NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(accum_f_ref_ref, ref_ref_vec_16u, vif_filt_s0[0]);
            NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(accum_f_dis_dis, dis_dis_vec_16u, vif_filt_s0[0]);

            uint32x4_t accum_f_ref_dis_0_l = vmulq_u32(accum_f_dis_0_l, vmovl_u16(vget_low_u16(ref_vec_16u_0)));
            uint32x4_t accum_f_ref_dis_0_h = vmulq_u32(accum_f_dis_0_h, vmovl_high_u16(ref_vec_16u_0));

            uint32x4_t accum_f_ref_dis_1_l = vmulq_u32(accum_f_dis_1_l, vmovl_u16(vget_low_u16(ref_vec_16u_1)));
            uint32x4_t accum_f_ref_dis_1_h = vmulq_u32(accum_f_dis_1_h, vmovl_high_u16(ref_vec_16u_1));

            const uint8_t *pp_ref = p_ref + buf.stride;
            const uint8_t *pp_dis = p_dis + buf.stride;
            for (unsigned int fi = 1; fi < fwidth; ++fi, pp_ref += buf.stride, pp_dis += buf.stride)
            {
                NEON_FILTER_LOAD_U8X8_MOVE_TO_U16X8_AND_SQR_LU2(ref_vec_8u, ref_vec_16u, ref_ref_vec_16u, pp_ref);
                NEON_FILTER_LOAD_U8X8_MOVE_TO_U16X8_AND_SQR_LU2(dis_vec_8u, dis_vec_16u, dis_dis_vec_16u, pp_dis);

                NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI_LU2(f_dis, dis_vec_16u, vif_filt_s0[fi]);

                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LU2(accum_f_ref, ref_vec_16u, vif_filt_s0[fi]);
                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LU2(accum_f_dis, dis_vec_16u, vif_filt_s0[fi]);
                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LU2(accum_f_ref_ref, ref_ref_vec_16u, vif_filt_s0[fi]);
                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI_LU2(accum_f_dis_dis, dis_dis_vec_16u, vif_filt_s0[fi]);

                accum_f_ref_dis_0_l = vmlaq_u32(accum_f_ref_dis_0_l, f_dis_0_l, vmovl_u16(vget_low_u16(ref_vec_16u_0)));
                accum_f_ref_dis_0_h = vmlaq_u32(accum_f_ref_dis_0_h, f_dis_0_h, vmovl_high_u16(ref_vec_16u_0));
                accum_f_ref_dis_1_l = vmlaq_u32(accum_f_ref_dis_1_l, f_dis_1_l, vmovl_u16(vget_low_u16(ref_vec_16u_1)));
                accum_f_ref_dis_1_h = vmlaq_u32(accum_f_ref_dis_1_h, f_dis_1_h, vmovl_high_u16(ref_vec_16u_1));
            }
            NEON_FILTER_OFFSET_SHIFT_STORE_U32X4_HI_LO_LU2(accum_f_ref, offset_vec_v, shift_vec_v, buf.tmp.mu1 + j);
            NEON_FILTER_OFFSET_SHIFT_STORE_U32X4_HI_LO_LU2(accum_f_dis, offset_vec_v, shift_vec_v, buf.tmp.mu2 + j);

            vst1q_u32(buf.tmp.ref + j, accum_f_ref_ref_0_l);
            vst1q_u32(buf.tmp.ref + j + 4, accum_f_ref_ref_0_h);
            vst1q_u32(buf.tmp.ref + j + 8, accum_f_ref_ref_1_l);
            vst1q_u32(buf.tmp.ref + j + 12, accum_f_ref_ref_1_h);

            vst1q_u32(buf.tmp.dis + j, accum_f_dis_dis_0_l);
            vst1q_u32(buf.tmp.dis + j + 4, accum_f_dis_dis_0_h);
            vst1q_u32(buf.tmp.dis + j + 8, accum_f_dis_dis_1_l);
            vst1q_u32(buf.tmp.dis + j + 12, accum_f_dis_dis_1_h);

            vst1q_u32(buf.tmp.ref_dis + j, accum_f_ref_dis_0_l);
            vst1q_u32(buf.tmp.ref_dis + j + 4, accum_f_ref_dis_0_h);
            vst1q_u32(buf.tmp.ref_dis + j + 8, accum_f_ref_dis_1_l);
            vst1q_u32(buf.tmp.ref_dis + j + 12, accum_f_ref_dis_1_h);
        }

        // Scalar code for Vertical leftover.
        for (; j < w; ++j)
        {
            uint32_t accum_mu1 = 128;
            uint32_t accum_mu2 = 128;
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            uint32_t accum_ref_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi)
            {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s0[fi];
                const uint8_t *ref = (uint8_t *)buf.ref;
                const uint8_t *dis = (uint8_t *)buf.dis;
                uint16_t imgcoeff_ref = ref[ii_check * buf.stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * buf.stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint32_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint32_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint32_t)imgcoeff_dis;
            }
            buf.tmp.mu1[j] = accum_mu1 >> 8;
            buf.tmp.mu2[j] = accum_mu2 >> 8;
            buf.tmp.ref[j] = accum_ref;
            buf.tmp.dis[j] = accum_dis;
            buf.tmp.ref_dis[j] = accum_ref_dis;
        }

        PADDING_SQ_DATA(buf, w, fwidth / 2);

        // HORIZONTAL
        uint32_t *pMul1 = (uint32_t *)buf.tmp.mu1 - (fwidth / 2);
        uint32_t *pMul2 = (uint32_t *)buf.tmp.mu2 - (fwidth / 2);
        uint32_t *pRef = (uint32_t *)buf.tmp.ref - (fwidth / 2);
        uint32_t *pDis = (uint32_t *)buf.tmp.dis - (fwidth / 2);
        uint32_t *pRefDis = (uint32_t *)buf.tmp.ref_dis - (fwidth / 2);

        j = 0;
        for (; j < uiw7; j += 8, pMul1 += 8, pMul2 += 8, pDis += 8, pRef += 8, pRefDis += 8)
        {
            uint32x4_t mul1_vec_u32_0 = vld1q_u32(pMul1);
            uint32x4_t mul2_vec_u32_0 = vld1q_u32(pMul2);
            uint32x4_t ref_vec_u32_0 = vld1q_u32(pRef);
            uint32x4_t dis_vec_u32_0 = vld1q_u32(pDis);
            uint32x4_t ref_dis_vec_u32_0 = vld1q_u32(pRefDis);

            uint32x4_t mul1_vec_u32_1 = vld1q_u32(pMul1 + 4);
            uint32x4_t mul2_vec_u32_1 = vld1q_u32(pMul2 + 4);
            uint32x4_t ref_vec_u32_1 = vld1q_u32(pRef + 4);
            uint32x4_t dis_vec_u32_1 = vld1q_u32(pDis + 4);
            uint32x4_t ref_dis_vec_u32_1 = vld1q_u32(pRefDis + 4);

            uint32x4_t accum_mu1_0 = vmulq_n_u32(mul1_vec_u32_0, vif_filt_s0[0]);
            uint32x4_t accum_mu2_0 = vmulq_n_u32(mul2_vec_u32_0, vif_filt_s0[0]);
            uint32x4_t accum_mu1_1 = vmulq_n_u32(mul1_vec_u32_1, vif_filt_s0[0]);
            uint32x4_t accum_mu2_1 = vmulq_n_u32(mul2_vec_u32_1, vif_filt_s0[0]);

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LU2(accum_ref, offset_vec_h, ref_vec_u32, vif_filt_s0[0]);
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LU2(accum_dis, offset_vec_h, dis_vec_u32, vif_filt_s0[0]);
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LU2(accum_ref_dis, offset_vec_h, ref_dis_vec_u32, vif_filt_s0[0]);

            for (unsigned fj = 1; fj < fwidth; ++fj)
            {
                mul1_vec_u32_0 = vld1q_u32(pMul1 + fj);
                mul2_vec_u32_0 = vld1q_u32(pMul2 + fj);
                ref_vec_u32_0 = vld1q_u32(pRef + fj);
                dis_vec_u32_0 = vld1q_u32(pDis + fj);
                ref_dis_vec_u32_0 = vld1q_u32(pRefDis + fj);
                mul1_vec_u32_1 = vld1q_u32(pMul1 + 4 + fj);
                mul2_vec_u32_1 = vld1q_u32(pMul2 + 4 + fj);
                ref_vec_u32_1 = vld1q_u32(pRef + 4 + fj);
                dis_vec_u32_1 = vld1q_u32(pDis + 4 + fj);
                ref_dis_vec_u32_1 = vld1q_u32(pRefDis + 4 + fj);

                accum_mu1_0 = vmlaq_n_u32(accum_mu1_0, mul1_vec_u32_0, vif_filt_s0[fj]);
                accum_mu2_0 = vmlaq_n_u32(accum_mu2_0, mul2_vec_u32_0, vif_filt_s0[fj]);
                accum_mu1_1 = vmlaq_n_u32(accum_mu1_1, mul1_vec_u32_1, vif_filt_s0[fj]);
                accum_mu2_1 = vmlaq_n_u32(accum_mu2_1, mul2_vec_u32_1, vif_filt_s0[fj]);

                NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI_LU2(accum_ref, ref_vec_u32, vif_filt_s0[fj]);
                NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI_LU2(accum_dis, dis_vec_u32, vif_filt_s0[fj]);
                NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI_LU2(accum_ref_dis, ref_dis_vec_u32, vif_filt_s0[fj]);
            }

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu1_sq_vec_l, vdupq_n_u64(2147483648), accum_mu1_0, accum_mu1_0, vdupq_n_s64(-32));
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu1_sq_vec_h, vdupq_n_u64(2147483648), accum_mu1_1, accum_mu1_1, vdupq_n_s64(-32));

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu2_sq_vec_l, vdupq_n_u64(2147483648), accum_mu2_0, accum_mu2_0, vdupq_n_s64(-32));
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu2_sq_vec_h, vdupq_n_u64(2147483648), accum_mu2_1, accum_mu2_1, vdupq_n_s64(-32));

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu1_mu2_sq_vec_l, vdupq_n_u64(2147483648), accum_mu1_0, accum_mu2_0, vdupq_n_s64(-32));
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu1_mu2_sq_vec_h, vdupq_n_u64(2147483648), accum_mu1_1, accum_mu2_1, vdupq_n_s64(-32));

            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_ref_0, shift_vec_h, xx_filt_vec_l);
            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_ref_1, shift_vec_h, xx_filt_vec_h);

            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_dis_0, shift_vec_h, yy_filt_vec_l);
            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_dis_1, shift_vec_h, yy_filt_vec_h);

            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_ref_dis_0, shift_vec_h, xy_filt_vec_l);
            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_ref_dis_1, shift_vec_h, xy_filt_vec_h);

            int32x4_t sigma1_sq_vec_l = vreinterpretq_s32_u32(vsubq_u32(xx_filt_vec_l,mu1_sq_vec_l));
            int32x4_t sigma1_sq_vec_h = vreinterpretq_s32_u32(vsubq_u32(xx_filt_vec_h,mu1_sq_vec_h));

            int32x4_t sigma2_sq_vec_l = vreinterpretq_s32_u32(vsubq_u32(yy_filt_vec_l,mu2_sq_vec_l));
            int32x4_t sigma2_sq_vec_h = vreinterpretq_s32_u32(vsubq_u32(yy_filt_vec_h,mu2_sq_vec_h));

            int32x4_t sigma12_vec_l = vreinterpretq_s32_u32(vsubq_u32(xy_filt_vec_l,mu1_mu2_sq_vec_l));
            int32x4_t sigma12_vec_h = vreinterpretq_s32_u32(vsubq_u32(xy_filt_vec_h,mu1_mu2_sq_vec_h));

            vst1q_s32(xx,       sigma1_sq_vec_l);
            vst1q_s32(xx + 4,   sigma1_sq_vec_h);

            vst1q_s32(yy,       sigma2_sq_vec_l);
            vst1q_s32(yy + 4,   sigma2_sq_vec_h);

            vst1q_s32(xy,       sigma12_vec_l);
            vst1q_s32(xy + 4,   sigma12_vec_h);


            for (unsigned int b = 0; b < 8; b++) {
                int32_t sigma1_sq = xx[b];
                int32_t sigma2_sq = yy[b];
                int32_t sigma12 = xy[b];

                if (sigma1_sq >= sigma_nsq) {
                    /**
                    * log values are taken from the look-up table generated by
                    * log_generate() function which is called in integer_combo_threadfunc
                    * den_val in float is log2(1 + sigma1_sq/2)
                    * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                    * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                    * x because best 16 bits are taken
                    */
                    accum_den_log += log2_32(log2_table, sigma_nsq + sigma1_sq) - 2048 * 17;

                    if (sigma12 > 0 && sigma2_sq > 0)
                    {
                        // num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
                        /**
                        * In floating-point numerator = log2((1.0f + (g * g * sigma1_sq)/(sv_sq + sigma_nsq))
                        *
                        * In Fixed-point the above is converted to
                        * numerator = log2((sv_sq + sigma_nsq)+(g * g * sigma1_sq))- log2(sv_sq + sigma_nsq)
                        */

                        const double eps = 65536 * 1.0e-10;
                        double g = sigma12 / (sigma1_sq + eps); // this epsilon can go away
                        int32_t sv_sq = sigma2_sq - g * sigma12;

                        sv_sq = (uint32_t)(MAX(sv_sq, 0));

                        g = MIN(g, vif_enhn_gain_limit);

                        uint32_t numer1 = (sv_sq + sigma_nsq);
                        int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
                        accum_num_log += log2_64(log2_table, numer1_tmp) - log2_64(log2_table, numer1);
                    }
                }
                else {
                    accum_num_non_log += sigma2_sq;
                    accum_den_non_log++;
                }
            }
        }
    }
    num[0] = accum_num_log / 2048.0 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 + accum_den_non_log;
}

void vif_statistic_16_neon(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h, int bpc, int scale)
{
    const unsigned int uiw7 = (w > 7 ? w - 7 : 0);
    const unsigned int fwidth = vif_filter1d_width[scale];
    const uint16_t *vif_filt_s = vif_filter1d_table[scale];
    VifBuffer buf = s->buf;
    uint16_t *log2_table = s->log2_table;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;

    int32_t add_shift_round_HP, shift_HP;
    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;
    if (scale == 0)
    {
        shift_HP = 16;
        add_shift_round_HP = 32768;
        shift_VP = bpc;
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP_sq = (bpc - 8) * 2;
        add_shift_round_VP_sq = (bpc == 8) ? 0 : 1 << (shift_VP_sq - 1);
    }
    else
    {
        shift_HP = 16;
        add_shift_round_HP = 32768;
        shift_VP = 16;
        add_shift_round_VP = 32768;
        shift_VP_sq = 16;
        add_shift_round_VP_sq = 32768;
    }

    const uint32x4_t add_shift_round_VP_vec = vdupq_n_u32(add_shift_round_VP);
    const int32x4_t shift_VP_vec = vdupq_n_s32(-shift_VP);
    const uint64x2_t add_shift_round_VP_sq_vec = vdupq_n_u64(add_shift_round_VP_sq);
    const int64x2_t shift_VP_sq_vec = vdupq_n_s64(-shift_VP_sq);

    const uint64x2_t add_shift_round_HP_vec = vdupq_n_u64(add_shift_round_HP);
    const int64x2_t shift_vec_HP = vdupq_n_s64(-shift_HP);

    const uint16_t *ref = (uint16_t *)buf.ref;
    const uint16_t *dis = (uint16_t *)buf.dis;

    const ptrdiff_t stride_16 = buf.stride / sizeof(uint16_t);
    const ptrdiff_t stride_32 = buf.stride_32 / sizeof(uint32_t);
    ptrdiff_t i_dst_stride = 0;
    int32_t xx[8], yy[8], xy[8];
    int64_t accum_num_log = 0.0;
    int64_t accum_den_log = 0.0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;
    static const int32_t sigma_nsq = 65536 << 1;

    for (unsigned i = 0; i < h; ++i, i_dst_stride += stride_32)
    {
        int ii = i - fwidth / 2;
        const uint16_t *p_ref = ref + ii * stride_16;
        const uint16_t *p_dis = dis + ii * stride_16;

        // VERTICAL 
        unsigned int j = 0;
        for (; j < uiw7; j += 8, p_ref += 8, p_dis += 8)
        {
            NEON_FILTER_LOAD_U16X8_AND_MOVE_TO_U32X4_AND_SQR(ref_vec_16u, ref_vec_32u, ref_ref_vec, p_ref);
            NEON_FILTER_LOAD_U16X8_AND_MOVE_TO_U32X4_AND_SQR(dis_vec_16u, dis_vec_32u, dis_dis_vec, p_dis);

            NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_f_ref, ref_vec_16u, vif_filt_s[0]);
            NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI(accum_f_dis, dis_vec_16u, vif_filt_s[0]);

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LH(accum_f_ref_ref, add_shift_round_VP_sq_vec, ref_ref_vec, vif_filt_s[0]);
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LH(accum_f_dis_dis, add_shift_round_VP_sq_vec, dis_dis_vec, vif_filt_s[0]);
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_LO_HI(accum_f_ref_dis_l, add_shift_round_VP_sq_vec, accum_f_dis_l, ref_vec_32u_l);
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_LO_HI(accum_f_ref_dis_h, add_shift_round_VP_sq_vec, accum_f_dis_h, ref_vec_32u_h);

            const uint16_t *pp_ref = p_ref + stride_16;
            const uint16_t *pp_dis = p_dis + stride_16;
            for (unsigned fi = 1; fi < fwidth; ++fi, pp_ref += stride_16, pp_dis += stride_16)
            {
                NEON_FILTER_LOAD_U16X8_AND_MOVE_TO_U32X4_AND_SQR(ref_vec_16u, ref_vec_32u, ref_ref_vec, pp_ref);
                NEON_FILTER_LOAD_U16X8_AND_MOVE_TO_U32X4_AND_SQR(dis_vec_16u, dis_vec_32u, dis_dis_vec, pp_dis);

                NEON_FILTER_INSTANCE_U32X4_NO_INIT_MULL_U16X4_WITH_CONST_LO_HI(f_dis, dis_vec_16u, vif_filt_s[fi]);

                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI(accum_f_ref, ref_vec_16u, vif_filt_s[fi]);
                NEON_FILTER_UPDATE_ACCUM_U32X4_WITH_CONST_LO_HI(accum_f_dis, dis_vec_16u, vif_filt_s[fi]);

                NEON_FILTER_ACCUM_LO_HI_LH_U64X2_WITH_CONST_LH(accum_f_ref_ref, ref_ref_vec, vif_filt_s[fi]);
                NEON_FILTER_ACCUM_LO_HI_LH_U64X2_WITH_CONST_LH(accum_f_dis_dis, dis_dis_vec, vif_filt_s[fi]);

                NEON_FILTER_UPDATE_U64X2_ACCUM_LO_HI(accum_f_ref_dis_l, f_dis_l, ref_vec_32u_l);
                NEON_FILTER_UPDATE_U64X2_ACCUM_LO_HI(accum_f_ref_dis_h, f_dis_h, ref_vec_32u_h);
            }

            NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_f_ref_l, add_shift_round_VP_vec, shift_VP_vec, buf.tmp.mu1 + j);
            NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_f_ref_h, add_shift_round_VP_vec, shift_VP_vec, buf.tmp.mu1 + j + 4);

            NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_f_dis_l, add_shift_round_VP_vec, shift_VP_vec, buf.tmp.mu2 + j);
            NEON_FILTER_OFFSET_SHIFT_STORE_U32X4(accum_f_dis_h, add_shift_round_VP_vec, shift_VP_vec, buf.tmp.mu2 + j + 4);

            NEON_FILTER_SHIFT_UNZIP_STORE_U64X2_TO_U32X4_LO_HI(accum_f_ref_ref_l, shift_VP_sq_vec, buf.tmp.ref + j);
            NEON_FILTER_SHIFT_UNZIP_STORE_U64X2_TO_U32X4_LO_HI(accum_f_ref_ref_h, shift_VP_sq_vec, buf.tmp.ref + j + 4);

            NEON_FILTER_SHIFT_UNZIP_STORE_U64X2_TO_U32X4_LO_HI(accum_f_dis_dis_l, shift_VP_sq_vec, buf.tmp.dis + j);
            NEON_FILTER_SHIFT_UNZIP_STORE_U64X2_TO_U32X4_LO_HI(accum_f_dis_dis_h, shift_VP_sq_vec, buf.tmp.dis + j + 4);

            NEON_FILTER_SHIFT_UNZIP_STORE_U64X2_TO_U32X4_LO_HI(accum_f_ref_dis_l, shift_VP_sq_vec, buf.tmp.ref_dis + j);
            NEON_FILTER_SHIFT_UNZIP_STORE_U64X2_TO_U32X4_LO_HI(accum_f_ref_dis_h, shift_VP_sq_vec, buf.tmp.ref_dis + j + 4);
        }

        // Scalar code for Vertical leftover.
        for (; j < w; ++j)
        {
            uint32_t accum_mu1 = add_shift_round_VP;
            uint32_t accum_mu2 = add_shift_round_VP;
            uint64_t accum_ref = add_shift_round_VP_sq;
            uint64_t accum_dis = add_shift_round_VP_sq;
            uint64_t accum_ref_dis = add_shift_round_VP_sq;
            for (unsigned fi = 0; fi < fwidth; ++fi)
            {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s[fi];
                const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
                uint16_t *ref = (uint16_t *)buf.ref;
                uint16_t *dis = (uint16_t *)buf.dis;
                uint16_t imgcoeff_ref = ref[ii_check * stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint64_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint64_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint64_t)imgcoeff_dis;
            }
            buf.tmp.mu1[j] = (uint16_t)(accum_mu1 >> shift_VP);
            buf.tmp.mu2[j] = (uint16_t)(accum_mu2 >> shift_VP);
            buf.tmp.ref[j] = (uint32_t)(accum_ref >> shift_VP_sq);
            buf.tmp.dis[j] = (uint32_t)(accum_dis >> shift_VP_sq);
            buf.tmp.ref_dis[j] = (uint32_t)(accum_ref_dis >> shift_VP_sq);
        }

        PADDING_SQ_DATA(buf, w, fwidth / 2);

        // HORIZONTAL
        uint32_t *pMul1 = (uint32_t *)buf.tmp.mu1 - (fwidth / 2);
        uint32_t *pMul2 = (uint32_t *)buf.tmp.mu2 - (fwidth / 2);
        uint32_t *pRef = (uint32_t *)buf.tmp.ref - (fwidth / 2);
        uint32_t *pDis = (uint32_t *)buf.tmp.dis - (fwidth / 2);
        uint32_t *pRefDis = (uint32_t *)buf.tmp.ref_dis - (fwidth / 2);

        j = 0;
        for (; j < uiw7; j += 8, pMul1 += 8, pMul2 += 8, pDis += 8, pRef += 8, pRefDis += 8)
        {
            uint32x4_t mul1_vec_u32_0 = vld1q_u32(pMul1);
            uint32x4_t mul2_vec_u32_0 = vld1q_u32(pMul2);
            uint32x4_t ref_vec_u32_0 = vld1q_u32(pRef);
            uint32x4_t dis_vec_u32_0 = vld1q_u32(pDis);
            uint32x4_t ref_dis_vec_u32_0 = vld1q_u32(pRefDis);

            uint32x4_t mul1_vec_u32_1 = vld1q_u32(pMul1 + 4);
            uint32x4_t mul2_vec_u32_1 = vld1q_u32(pMul2 + 4);
            uint32x4_t ref_vec_u32_1 = vld1q_u32(pRef + 4);
            uint32x4_t dis_vec_u32_1 = vld1q_u32(pDis + 4);
            uint32x4_t ref_dis_vec_u32_1 = vld1q_u32(pRefDis + 4);

            uint32x4_t accum_mu1_0 = vmulq_n_u32(mul1_vec_u32_0, vif_filt_s[0]);
            uint32x4_t accum_mu2_0 = vmulq_n_u32(mul2_vec_u32_0, vif_filt_s[0]);
            uint32x4_t accum_mu1_1 = vmulq_n_u32(mul1_vec_u32_1, vif_filt_s[0]);
            uint32x4_t accum_mu2_1 = vmulq_n_u32(mul2_vec_u32_1, vif_filt_s[0]);

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LU2(accum_ref, add_shift_round_HP_vec, ref_vec_u32, vif_filt_s[0]);
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LU2(accum_dis, add_shift_round_HP_vec, dis_vec_u32, vif_filt_s[0]);
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_U32X2_WITH_CONST_LO_HI_LU2(accum_ref_dis, add_shift_round_HP_vec, ref_dis_vec_u32, vif_filt_s[0]);

            for (unsigned fj = 1; fj < fwidth; ++fj)
            {
                mul1_vec_u32_0 = vld1q_u32(pMul1 + fj);
                mul2_vec_u32_0 = vld1q_u32(pMul2 + fj);
                ref_vec_u32_0 = vld1q_u32(pRef + fj);
                dis_vec_u32_0 = vld1q_u32(pDis + fj);
                ref_dis_vec_u32_0 = vld1q_u32(pRefDis + fj);
                mul1_vec_u32_1 = vld1q_u32(pMul1 + 4 + fj);
                mul2_vec_u32_1 = vld1q_u32(pMul2 + 4 + fj);
                ref_vec_u32_1 = vld1q_u32(pRef + 4 + fj);
                dis_vec_u32_1 = vld1q_u32(pDis + 4 + fj);
                ref_dis_vec_u32_1 = vld1q_u32(pRefDis + 4 + fj);

                accum_mu1_0 = vmlaq_n_u32(accum_mu1_0, mul1_vec_u32_0, vif_filt_s[fj]);
                accum_mu2_0 = vmlaq_n_u32(accum_mu2_0, mul2_vec_u32_0, vif_filt_s[fj]);
                accum_mu1_1 = vmlaq_n_u32(accum_mu1_1, mul1_vec_u32_1, vif_filt_s[fj]);
                accum_mu2_1 = vmlaq_n_u32(accum_mu2_1, mul2_vec_u32_1, vif_filt_s[fj]);

                NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI_LU2(accum_ref, ref_vec_u32, vif_filt_s[fj]);
                NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI_LU2(accum_dis, dis_vec_u32, vif_filt_s[fj]);
                NEON_FILTER_UPDATE_ACCUM_U64X2_WITH_CONST_LO_HI_LU2(accum_ref_dis, ref_dis_vec_u32, vif_filt_s[fj]);
            }

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu1_sq_vec_l, vdupq_n_u64(2147483648), accum_mu1_0, accum_mu1_0, vdupq_n_s64(-32));
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu1_sq_vec_h, vdupq_n_u64(2147483648), accum_mu1_1, accum_mu1_1, vdupq_n_s64(-32));

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu2_sq_vec_l, vdupq_n_u64(2147483648), accum_mu2_0, accum_mu2_0, vdupq_n_s64(-32));
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu2_sq_vec_h, vdupq_n_u64(2147483648), accum_mu2_1, accum_mu2_1, vdupq_n_s64(-32));

            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu1_mu2_sq_vec_l, vdupq_n_u64(2147483648), accum_mu1_0, accum_mu2_0, vdupq_n_s64(-32));
            NEON_FILTER_INSTANCE_U64X2_INIT_MULL_SHIFT_UNZIP_U32X4_LO_HI(mu1_mu2_sq_vec_h, vdupq_n_u64(2147483648), accum_mu1_1, accum_mu2_1, vdupq_n_s64(-32));

            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_ref_0, shift_vec_HP, xx_filt_vec_l);
            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_ref_1, shift_vec_HP, xx_filt_vec_h);

            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_dis_0, shift_vec_HP, yy_filt_vec_l);
            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_dis_1, shift_vec_HP, yy_filt_vec_h);

            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_ref_dis_0, shift_vec_HP, xy_filt_vec_l);
            NEON_FILTER_SHIFT_UNZIP_U64X2_TO_U32X4_LO_HI(accum_ref_dis_1, shift_vec_HP , xy_filt_vec_h);

            int32x4_t sigma1_sq_vec_l = vreinterpretq_s32_u32(vsubq_u32(xx_filt_vec_l,mu1_sq_vec_l));
            int32x4_t sigma1_sq_vec_h = vreinterpretq_s32_u32(vsubq_u32(xx_filt_vec_h,mu1_sq_vec_h));

            int32x4_t sigma2_sq_vec_l = vreinterpretq_s32_u32(vsubq_u32(yy_filt_vec_l,mu2_sq_vec_l));
            int32x4_t sigma2_sq_vec_h = vreinterpretq_s32_u32(vsubq_u32(yy_filt_vec_h,mu2_sq_vec_h));

            int32x4_t sigma12_vec_l = vreinterpretq_s32_u32(vsubq_u32(xy_filt_vec_l,mu1_mu2_sq_vec_l));
            int32x4_t sigma12_vec_h = vreinterpretq_s32_u32(vsubq_u32(xy_filt_vec_h,mu1_mu2_sq_vec_h));

            vst1q_s32(xx,       sigma1_sq_vec_l);
            vst1q_s32(xx + 4,   sigma1_sq_vec_h);

            vst1q_s32(yy,       vmaxq_s32(vdupq_n_s32(0),sigma2_sq_vec_l));
            vst1q_s32(yy + 4,   vmaxq_s32(vdupq_n_s32(0),sigma2_sq_vec_h));

            vst1q_s32(xy,       sigma12_vec_l);
            vst1q_s32(xy + 4,   sigma12_vec_h);

            for (unsigned int b = 0; b < 8; b++) {
                int32_t sigma1_sq = xx[b];
                int32_t sigma2_sq = yy[b];
                int32_t sigma12 = xy[b];

                if (sigma1_sq >= sigma_nsq) {
                    /**
                    * log values are taken from the look-up table generated by
                    * log_generate() function which is called in integer_combo_threadfunc
                    * den_val in float is log2(1 + sigma1_sq/2)
                    * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                    * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                    * x because best 16 bits are taken
                    */
                    accum_den_log += log2_32(log2_table, sigma_nsq + sigma1_sq) - 2048 * 17;

                    if (sigma12 > 0 && sigma2_sq > 0)
                    {
                        // num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
                        /**
                        * In floating-point numerator = log2((1.0f + (g * g * sigma1_sq)/(sv_sq + sigma_nsq))
                        *
                        * In Fixed-point the above is converted to
                        * numerator = log2((sv_sq + sigma_nsq)+(g * g * sigma1_sq))- log2(sv_sq + sigma_nsq)
                        */

                        const double eps = 65536 * 1.0e-10;
                        double g = sigma12 / (sigma1_sq + eps); // this epsilon can go away
                        int32_t sv_sq = sigma2_sq - g * sigma12;

                        sv_sq = (uint32_t)(MAX(sv_sq, 0));

                        g = MIN(g, vif_enhn_gain_limit);

                        uint32_t numer1 = (sv_sq + sigma_nsq);
                        int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
                        accum_num_log += log2_64(log2_table, numer1_tmp) - log2_64(log2_table, numer1);
                    }
                }
                else {
                    accum_num_non_log += sigma2_sq;
                    accum_den_non_log++;
                }
            }
        }

        if (j != w)
        {
            VifResiduals residuals =
                vif_compute_line_residuals(s, j, w, scale);
            accum_num_log += residuals.accum_num_log;
            accum_den_log += residuals.accum_den_log;
            accum_num_non_log += residuals.accum_num_non_log;
            accum_den_non_log += residuals.accum_den_non_log;
        }
    }
    num[0] = accum_num_log / 2048.0 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 + accum_den_non_log;
}

