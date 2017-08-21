/*
 * Copyright (C) 2007 Marco Gerards <marco@gnu.org>
 * Copyright (C) 2009 David Conrad
 * Copyright (C) 2011 Jordi Ortiz
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Dirac Decoder
 * @author Marco Gerards <marco@gnu.org>, David Conrad, Jordi Ortiz <nenjordi@gmail.com>
 */

#include "avcodec.h"
#include "get_bits.h"
#include "internal.h"
#include "golomb.h"
#include "dirac_dwt.h"
#include "dirac_vlc.h"
#include "diractab.h"
#include "dirac.h"
#include "diracdsp.h"
#include "mpegvideo.h"

/**
 * The spec limits the number of wavelet decompositions to 4 for both
 * level 1 (VC-2) and 128 (long-gop default).
 * 5 decompositions is the maximum before >16-bit buffers are needed.
 * Schroedinger allows this for DD 9,7 and 13,7 wavelets only, limiting
 * the others to 4 decompositions (or 3 for the fidelity filter).
 *
 * We use this instead of MAX_DECOMPOSITIONS to save some memory.
 */
#define MAX_DWT_LEVELS 5

#define MAX_QUANT 255        /* max quant for VC-2 */
#define MAX_BLOCKSIZE 32     /* maximum xblen/yblen we support */

#define CALC_PADDING(size, depth)                       \
    (((size + (1 << depth) - 1) >> depth) << depth)

typedef struct SubBand {
    int level;
    int orientation;
    int stride; /* in bytes */
    int width;
    int height;
    int pshift;
    int quant;
    uint8_t *ibuf;

    /* for low delay */
    unsigned length;
    const uint8_t *coeff_data;
} SubBand;

typedef struct Plane {
    DWTPlane idwt;

    int width;
    int height;
    ptrdiff_t stride;

    SubBand band[MAX_DWT_LEVELS][4];
} Plane;

/* Used by Low Delay and High Quality profiles */
typedef struct DiracSlice {
    GetBitContext gb;
    int slice_x;
    int slice_y;
    int bytes;
} DiracSlice;

typedef struct DiracContext {
    AVCodecContext *avctx;
    DiracDSPContext diracdsp;
    DiracVersionInfo version;
    GetBitContext gb;
    AVDiracSeqHeader seq;
    int seen_sequence_header;
    int prev_pict_number;       /* detect discontinuities */
    int frame_number;           /* number of the next frame to display       */
    Plane plane[3];
    int chroma_x_shift;
    int chroma_y_shift;

    DiracGolombLUT *reader_ctx; /* Lookup table for the golomb reader        */

    int bit_depth;              /* bit depth                                 */
    int pshift;                 /* pixel shift = bit_depth > 8               */

    int field_coding;           /* fields instead of frames                  */
    int cur_field;              /* 0 -> progressive/top, 1 -> bottom         */
    enum AVPixelFormat prev_field_fmt;

    uint8_t *thread_buf;        /* Per-thread buffer for coefficient storage */
    int threads_num_buf;        /* Current # of buffers allocated            */
    int thread_buf_size;        /* Each thread has a buffer this size        */

    DiracSlice *slice_params_buf;
    int slice_params_num_buf;

    /* fragments */
    int is_fragment;
    int fragment_data_length;
    int fragment_slice_count;
    int fragment_slices_received;

    /* wavelet decoding */
    unsigned wavelet_depth;     /* depth of the IDWT                         */
    unsigned wavelet_idx;

    unsigned num_x;              /* number of horizontal slices              */
    unsigned num_y;              /* number of vertical slices                */

    uint8_t quant[MAX_DWT_LEVELS][4]; /* Quantization matrix */

    unsigned prefix_bytes;
    uint64_t size_scaler;

    int seq_buf_allocated_width;
    int seq_buf_allocated_height;
    int seq_buf_allocated_coding;
    enum AVPixelFormat seq_buf_allocated_fmt;

    AVFrame *dummy_frame, *prev_field, *current_picture;
    AVFrame fragment_picture;
} DiracContext;

static int alloc_sequence_buffers(DiracContext *s)
{
    int i, w, h, top_padding;

    /* todo: think more about this / use or set Plane here */
    for (i = 0; i < 3; i++) {
        int max_xblen = MAX_BLOCKSIZE >> (i ? s->chroma_x_shift : 0);
        int max_yblen = MAX_BLOCKSIZE >> (i ? s->chroma_y_shift : 0);
        w = s->seq.width  >> (i ? s->chroma_x_shift : 0);
        h = s->seq.height >> (i ? s->chroma_y_shift : 0);

        /* we allocate the max we support here since num decompositions can
         * change from frame to frame. Stride is aligned to 16 for SIMD, and
         * 1<<MAX_DWT_LEVELS top padding to avoid if(y>0) in arith decoding
         * MAX_BLOCKSIZE padding for MC: blocks can spill up to half of that
         * on each side */
        top_padding = FFMAX(1<<MAX_DWT_LEVELS, max_yblen/2);
        w = FFALIGN(CALC_PADDING(w, MAX_DWT_LEVELS), 8); /* FIXME: Should this be 16 for SSE??? */
        h = top_padding + CALC_PADDING(h, MAX_DWT_LEVELS) + max_yblen/2;

        s->plane[i].idwt.buf_base = av_mallocz_array((w+max_xblen), h * (2 << s->pshift));
        s->plane[i].idwt.tmp      = av_malloc_array((w+16), 2 << s->pshift);
        s->plane[i].idwt.buf      = s->plane[i].idwt.buf_base + (top_padding*w)*(2 << s->pshift);
        if (!s->plane[i].idwt.buf_base || !s->plane[i].idwt.tmp)
            return AVERROR(ENOMEM);
    }

    return 0;
}

static void free_sequence_buffers(DiracContext *s)
{
    int i;

    for (i = 0; i < 3; i++) {
        av_freep(&s->plane[i].idwt.buf_base);
        av_freep(&s->plane[i].idwt.tmp);
    }
    s->seq_buf_allocated_width  = 0;
    s->seq_buf_allocated_height = 0;
    s->seq_buf_allocated_fmt    = 0;
}

static av_cold int dirac_decode_init(AVCodecContext *avctx)
{
    DiracContext *s = avctx->priv_data;

    s->avctx = avctx;
    s->prev_pict_number = 0;
    s->current_picture = NULL;

    ff_diracdsp_init(&s->diracdsp);

    /* Initialize the golomb reader */
    ff_dirac_golomb_reader_init(&s->reader_ctx);

    s->thread_buf = NULL;
    s->threads_num_buf = -1;

    s->slice_params_buf = NULL;
    s->slice_params_num_buf = -1;

    s->dummy_frame = av_frame_alloc();

    return 0;
}

static void dirac_decode_flush(AVCodecContext *avctx)
{
    DiracContext *s = avctx->priv_data;
    free_sequence_buffers(s);
    s->seen_sequence_header = 0;
}

static av_cold int dirac_decode_end(AVCodecContext *avctx)
{
    DiracContext *s = avctx->priv_data;

    dirac_decode_flush(avctx);

    ff_dirac_golomb_reader_end(&s->reader_ctx);

    av_freep(&s->thread_buf);
    av_freep(&s->slice_params_buf);

    av_frame_free(&s->dummy_frame);

    return 0;
}

/* Records coordinates - where exactly to put the coefficients from each
 * slice to the big coefficients buffer, as well as how many coefficients there are */
typedef struct SliceCoeffs {
    int left;
    int top;
    int tot_h;
    int tot_v;
    int tot;
} SliceCoeffs;

/* Counts the total amount of coefficients per level per orientation per slice */
static int subband_coeffs(DiracContext *s, int x, int y, int p,
                          SliceCoeffs c[MAX_DWT_LEVELS])
{
    int level, coef = 0;
    for (level = 0; level < s->wavelet_depth; level++) {
        SliceCoeffs *o = &c[level];
        SubBand *b = &s->plane[p].band[level][3]; /* orientation doens't matter */
        o->top   = b->height * y / s->num_y;
        o->left  = b->width  * x / s->num_x;
        o->tot_h = ((b->width  * (x + 1)) / s->num_x) - o->left;
        o->tot_v = ((b->height * (y + 1)) / s->num_y) - o->top;
        o->tot   = o->tot_h*o->tot_v;
        coef    += o->tot * (4 - !!level);
    }
    return coef;
}

/**
 * VC-2 Specification ->
 * 13.5.3 hq_slice(sx,sy)
 */
static int decode_hq_slice(DiracContext *s, DiracSlice *slice, uint8_t *tmp_buf)
{
    int i, level, orientation, quant_idx;
    int qfactor[MAX_DWT_LEVELS][4], qoffset[MAX_DWT_LEVELS][4];
    GetBitContext *gb = &slice->gb;
    SliceCoeffs coeffs_num[MAX_DWT_LEVELS];

    /* Skip the prefix bytes and read the quantization index */
    skip_bits_long(gb, 8*s->prefix_bytes);
    quant_idx = get_bits(gb, 8);

    /* Very likely a corruption */
    if (quant_idx > DIRAC_MAX_QUANT_INDEX) {
        av_log(s->avctx, AV_LOG_ERROR, "Invalid quantization index - %i\n", quant_idx);
        return AVERROR_INVALIDDATA;
    }

    /* Slice quantization (slice_quantizers() in the specs) */
    for (level = 0; level < s->wavelet_depth; level++) {
        for (orientation = !!level; orientation < 4; orientation++) {
            const int quant = FFMAX(quant_idx - s->quant[level][orientation], 0);
            qfactor[level][orientation] = ff_dirac_qscale_tab[quant];
            qoffset[level][orientation] = ff_dirac_qoffset_intra_tab[quant] + 2;
        }
    }

    /* Luma + 2 Chroma planes */
    for (i = 0; i < 3; i++) {
        int coef_num, coef_par, off = 0;
        int64_t length = s->size_scaler*get_bits(gb, 8);
        int64_t bits_end = get_bits_count(gb) + 8*length;
        const uint8_t *addr = align_get_bits(gb);

        /* Corruption */
        if (length*8 > get_bits_left(gb)) {
            av_log(s->avctx, AV_LOG_ERROR, "end too far away\n");
            return AVERROR_INVALIDDATA;
        }

        /* Total coefficients in this plane + fill in coeffs_num structure */
        coef_num = subband_coeffs(s, slice->slice_x, slice->slice_y, i, coeffs_num);

        /* Runs the golomb decoder to decode the coefficients into the buffer */
        if (s->pshift)
            coef_par = ff_dirac_golomb_read_32bit(s->reader_ctx, addr,
                                                  length, tmp_buf, coef_num);
        else
            coef_par = ff_dirac_golomb_read_16bit(s->reader_ctx, addr,
                                                  length, tmp_buf, coef_num);

        /* Will memset to 0, as per the spec, any remaining coefficients if the
         * slice ends early to save bits on 0s */
        if (coef_num > coef_par) {
            const int start_b = coef_par * (1 << (s->pshift + 1));
            const int end_b   = coef_num * (1 << (s->pshift + 1));
            memset(&tmp_buf[start_b], 0, end_b - start_b);
        }

        for (level = 0; level < s->wavelet_depth; level++) {
            const SliceCoeffs *c = &coeffs_num[level];
            for (orientation = !!level; orientation < 4; orientation++) {
                const SubBand *b1 = &s->plane[i].band[level][orientation];
                uint8_t *buf = b1->ibuf + c->top * b1->stride + (c->left << (s->pshift + 1));

                /* Determines whether to use the C or the SIMD version of the
                 * dequant function, needs to do so because with small slice
                 * sizes you won't have enough coefficients in the buffer for SIMD.
                 * The C version is ran once a slice for a coeff or two at most,
                 * so no performance issues. Change to c->tot_h <= 4 for AVX2
                 * dequantization */
                const int qfunc = s->pshift + 2*(c->tot_h <= 2);

                /* Runs the dequant SIMD which puts the serial coefficients into
                 * the proper image coefficient buffer places */
                s->diracdsp.dequant_subband[qfunc](&tmp_buf[off], buf, b1->stride,
                                                   qfactor[level][orientation],
                                                   qoffset[level][orientation],
                                                   c->tot_v, c->tot_h);

                off += c->tot << (s->pshift + 1);
            }
        }

        /* Skip to the next plane */
        skip_bits_long(gb, bits_end - get_bits_count(gb));
    }

    return 0;
}

static int decode_hq_slice_row(AVCodecContext *avctx, void *arg, int jobnr, int threadnr)
{
    int i;
    DiracContext *s = avctx->priv_data;
    /* Starting slice */
    DiracSlice *slices = ((DiracSlice *)arg) + s->num_x*jobnr;
    /* Reason to use execute2 - the function knows which thread number it has */
    uint8_t *thread_buf = &s->thread_buf[s->thread_buf_size*threadnr];
    /* Loop over all horizontal slices in the row */
    for (i = 0; i < s->num_x; i++)
        decode_hq_slice(s, &slices[i], thread_buf); //TODO: use returned error code
    return 0;
}

/**
 * Dirac Specification ->
 * 13.5.1 low_delay_transform_data()
 */
static int decode_lowdelay(DiracContext *s)
{
    AVCodecContext *avctx = s->avctx;
    int i, slice_x, slice_y, bufsize, coef_buf_size, slice_num = 0, fragment_data_len, x_offset, y_offset, slice;
    int64_t bytes = 0;
    const uint8_t *buf;
    DiracSlice *slices;
    SliceCoeffs tmp[MAX_DWT_LEVELS];

    /* Reallocs the slices arguments buffer in case the number of slices change */
    if (s->fragment_slices_received == 0 && s->slice_params_num_buf != (s->num_x * s->num_y)) {
        s->slice_params_buf = av_realloc_f(s->slice_params_buf, s->num_x * s->num_y, sizeof(DiracSlice));
        if (!s->slice_params_buf) {
            av_log(s->avctx, AV_LOG_ERROR, "slice params buffer allocation failure\n");
            return AVERROR(ENOMEM);
        }
        s->slice_params_num_buf = s->num_x * s->num_y;
    }
    slices = s->slice_params_buf;

    /* Reallocs the thread buffer - the temporary storage for slice coefficients
    * for a single thread if the number of threads changes during runtime
    * which can definitely happen.
    * This is the maximum amount of coefficients in a slice in an entire frame
    * NB: number of coefficients in a slice in a frame is NOT constant and will
    * change depending on where the slice is, hence this is the maximum.
    * 8 becacuse that's how much the golomb reader could overread junk data
    * from another plane/slice at most, and 512 because SIMD */
    coef_buf_size = subband_coeffs(s, s->num_x - 1, s->num_y - 1, 0, tmp) + 8;
    coef_buf_size = (coef_buf_size << (1 + s->pshift)) + 512;

    if (s->threads_num_buf != avctx->thread_count ||
        s->thread_buf_size != coef_buf_size) {
        s->threads_num_buf  = avctx->thread_count;
        s->thread_buf_size  = coef_buf_size;
        s->thread_buf       = av_realloc_f(s->thread_buf, avctx->thread_count, s->thread_buf_size);
        if (!s->thread_buf) {
            av_log(s->avctx, AV_LOG_ERROR, "thread buffer allocation failure\n");
            return AVERROR(ENOMEM);
        }
    }

    align_get_bits(&s->gb);
    buf = s->gb.buffer + get_bits_count(&s->gb)/8;
    bufsize = get_bits_left(&s->gb);

    if (s->is_fragment) {
        fragment_data_len = get_bits_long(&s->gb, 16);
        if (fragment_data_len > bufsize) {
            av_log(s->avctx, AV_LOG_ERROR, "invalid fragment data length \n");
            return AVERROR_INVALIDDATA;
        }

        slice_num = get_bits_long(&s->gb, 16);
        if (!slice_num || s->fragment_slices_received + slice_num > s->num_x*s->num_y) {
            av_log(s->avctx, AV_LOG_ERROR, "invalid number of slices\n");
            return AVERROR_INVALIDDATA;
        }

        x_offset = get_bits_long(&s->gb, 16);
        y_offset = get_bits_long(&s->gb, 16);

        if (x_offset >= s->num_x || y_offset >= s->num_y) {
            av_log(s->avctx, AV_LOG_ERROR, "fragment slice offset (%d,%d) is invalid for slice dimensions (%dx%d)\n",
                    x_offset, y_offset, s->num_x, s->num_y);
            return AVERROR_INVALIDDATA;
        }

        /* byte_align and update buffer position before fragment_data */
        align_get_bits(&s->gb);
        buf = s->gb.buffer + get_bits_count(&s->gb)/8;
        bufsize = get_bits_left(&s->gb);

        for (slice = 0; slice < slice_num; slice++) {
            bytes = s->prefix_bytes + 1;
            for (i = 0; i < 3; i++) {
                if (bytes <= bufsize/8)
                    bytes += buf[bytes] * s->size_scaler + 1;
            }
            if (bytes >= INT_MAX || bytes*8 > bufsize) {
                av_log(s->avctx, AV_LOG_ERROR, "too many bytes\n"); //TODO: improve message
                return AVERROR_INVALIDDATA;
            }

            slices[slice].bytes   = bytes;
            slices[slice].slice_x = (y_offset*s->num_x + x_offset + slice) % s->num_x;
            slices[slice].slice_y = (y_offset*s->num_x + x_offset + slice) / s->num_x;
            /* Sets all the arguments and get bit contexts to the right addresses */
            init_get_bits(&slices[slice].gb, buf, bufsize);

            buf += bytes;
            if (bufsize/8 >= bytes)
                bufsize -= bytes*8;
            else
                bufsize = 0;

            decode_hq_slice(s, &slices[slice], s->thread_buf); //TODO: use returned error code
        }
        s->fragment_slices_received += slice_num;
    } else {
        /*[DIRAC_STD] 13.5.2 Slices. slice(sx,sy) */
        for (slice_y = 0; bufsize > 0 && slice_y < s->num_y; slice_y++) {
            for (slice_x = 0; bufsize > 0 && slice_x < s->num_x; slice_x++) {
                bytes = s->prefix_bytes + 1;
                for (i = 0; i < 3; i++) {
                    if (bytes <= bufsize/8)
                        bytes += buf[bytes] * s->size_scaler + 1;
                }
                if (bytes >= INT_MAX || bytes*8 > bufsize) {
                    av_log(s->avctx, AV_LOG_ERROR, "too many bytes\n");
                    return AVERROR_INVALIDDATA;
                }

                slices[slice_num].bytes   = bytes;
                slices[slice_num].slice_x = slice_x;
                slices[slice_num].slice_y = slice_y;
                /* Sets all the arguments and get bit contexts to the right addresses */
                init_get_bits(&slices[slice_num].gb, buf, bufsize);
                slice_num++;

                buf += bytes;
                if (bufsize/8 >= bytes)
                    bufsize -= bytes*8;
                else
                    bufsize = 0;
            }
        }

        /* Very likely to detect overflows */
        if (s->num_x*s->num_y != slice_num) {
            av_log(s->avctx, AV_LOG_ERROR, "too few slices\n");
            return AVERROR_INVALIDDATA;
        }

        /* Runs decode_hq_slice_row to decode the slices in a row per thread */
        avctx->execute2(avctx, decode_hq_slice_row, slices, NULL, s->num_y);
    }

    return 0;
}

static void init_planes(DiracContext *s)
{
    int i, w, h, level, orientation;

    /* Does the major per plane buffer allocation and sets up the coefficient
     * strides and orientations */
    for (i = 0; i < 3; i++) {
        Plane *p = &s->plane[i];

        p->width       = s->seq.width  >> (i ? s->chroma_x_shift : 0);
        p->height      = s->seq.height >> (i ? s->chroma_y_shift : 0);
        p->idwt.width  = w = CALC_PADDING(p->width , s->wavelet_depth);
        p->idwt.height = h = CALC_PADDING(p->height, s->wavelet_depth);
        p->idwt.stride = FFALIGN(p->idwt.width, 8) << (1 + s->pshift);

        for (level = s->wavelet_depth-1; level >= 0; level--) {
            w = w>>1;
            h = h>>1;
            for (orientation = !!level; orientation < 4; orientation++) {
                SubBand *b = &p->band[level][orientation];

                b->pshift = s->pshift;
                b->ibuf   = p->idwt.buf;
                b->level  = level;
                b->stride = p->idwt.stride << (s->wavelet_depth - level);
                b->width  = w;
                b->height = h;
                b->orientation = orientation;

                if (orientation & 1)
                    b->ibuf += w << (1+b->pshift);
                if (orientation > 1)
                    b->ibuf += (b->stride >> 1);
            }
        }
    }
}

/**
 * Dirac Specification ->
 * 11.3 Wavelet transform data. wavelet_transform()
 */
static int dirac_unpack_idwt_params(DiracContext *s)
{
    GetBitContext *gb = &s->gb;
    int i, level;
    int tmp, asym_transform_index_flag, asym_transform_flag;

#define CHECKEDREAD(dst, cond, errmsg) \
    tmp = get_interleaved_ue_golomb(gb); \
    if (cond) { \
        av_log(s->avctx, AV_LOG_ERROR, errmsg); \
        return AVERROR_INVALIDDATA; \
    }\
    dst = tmp;

    align_get_bits(gb);

    /*[DIRAC_STD] 11.3.1 Transform parameters. transform_parameters() */
    CHECKEDREAD(s->wavelet_idx, tmp > 6, "wavelet_idx is too big\n")

    CHECKEDREAD(s->wavelet_depth, tmp > MAX_DWT_LEVELS || tmp < 1, "invalid number of DWT decompositions\n")

    if (s->version.major >= 3) {
        asym_transform_index_flag = get_bits1(gb);
        if (asym_transform_index_flag)
            get_interleaved_ue_golomb(gb);
        
        asym_transform_flag = get_bits1(gb);
        if (asym_transform_flag)
            get_interleaved_ue_golomb(gb);
        
        if (asym_transform_index_flag || asym_transform_flag)
            avpriv_request_sample(s->avctx, "Asymmetric transform");
    }

    /* Min slice size is 8 pixels, so it's a sane limit */
    CHECKEDREAD(s->num_x, (tmp <= 0 || (tmp > (s->avctx->width /8))), "Invalid number of horizontal slices\n");
    CHECKEDREAD(s->num_y, (tmp <= 0 || (tmp > (s->avctx->height/8))), "Invalid number of vertical slices\n");

    s->prefix_bytes = get_interleaved_ue_golomb(gb);

    CHECKEDREAD(s->size_scaler, tmp <= 0, "Invalid slice scaler\n");

    if (s->prefix_bytes >= INT_MAX / 8) {
        av_log(s->avctx,AV_LOG_ERROR,"too many prefix bytes\n");
        return AVERROR_INVALIDDATA;
    }

    /* [DIRAC_STD] 11.3.5 Quantisation matrices (low-delay syntax). quant_matrix() */
    if (get_bits1(gb)) {
        av_log(s->avctx,AV_LOG_DEBUG,"Low Delay: Has Custom Quantization Matrix!\n");
        /* custom quantization matrix */
        s->quant[0][0] = get_interleaved_ue_golomb(gb);
        for (level = 0; level < s->wavelet_depth; level++) {
            s->quant[level][1] = get_interleaved_ue_golomb(gb);
            s->quant[level][2] = get_interleaved_ue_golomb(gb);
            s->quant[level][3] = get_interleaved_ue_golomb(gb);
        }
    } else {
        if (s->wavelet_depth > 4) {
            av_log(s->avctx,AV_LOG_ERROR,"Mandatory custom low delay matrix missing for depth %d\n", s->wavelet_depth);
            return AVERROR_INVALIDDATA;
        }
        /* default quantization matrix */
        for (level = 0; level < s->wavelet_depth; level++)
            for (i = 0; i < 4; i++) {
                s->quant[level][i] = ff_dirac_default_qmat[s->wavelet_idx][level][i];
                /* haar with no shift differs for different depths */
                if (s->wavelet_idx == 3)
                    s->quant[level][i] += 4*(s->wavelet_depth - 1 - level);
            }
    }

    return 0;
}

static int idwt_plane(AVCodecContext *avctx, void *arg, int jobnr, int threadnr)
{
    int y;
    DWTContext d;
    DiracContext *s   = avctx->priv_data;
    Plane *p          = &s->plane[jobnr];
    uint8_t *frame    = s->current_picture->data[jobnr];
    const int idx     = (s->bit_depth - 8) >> 1;
    const int ostride = p->stride << s->field_coding;

    /* Interleaves the fields */
    frame += s->cur_field * p->stride;

    /* Sets up the DWT context pointers */
    ff_spatial_idwt_init(&d, &p->idwt, s->wavelet_idx + 2, s->wavelet_depth, s->bit_depth);

    /* Does 16 lines at a time -
     * Could reduce the latency by returning 16 lines instead of an entire frame
     * and even further if you decode and idwt a row of slices at once */
    for (y = 0; y < p->height; y += 16) {
        ff_spatial_idwt_slice2(&d, y+16); /* decode */
        /* After doing the 16-line iDWT run the signed->unsigned SIMD */
        s->diracdsp.put_signed_rect_clamped[idx](frame + y*ostride,
                                                 ostride,
                                                 p->idwt.buf + y*p->idwt.stride,
                                                 p->idwt.stride, p->width, 16);
    }

    return 0;
}

/**
 * Dirac Specification ->
 * 13.0 Transform data syntax. transform_data()
 */
static int dirac_decode_frame_internal(DiracContext *s)
{
    int ret;

    /* Does the actual coefficient decoding */
    if ((ret = decode_lowdelay(s)) < 0)
        return ret;

    if (!s->is_fragment) {
        /* Does the iDWT on the 3 planes in parallel, not in git master since
        * the MC depends on doing it serially */
        s->avctx->execute2(s->avctx, idwt_plane, NULL, NULL, 3);
    }

    return 0;
}

static int get_buffer_with_edge(AVCodecContext *avctx, AVFrame *f, int flags)
{
    int ret;

    /* Pads the allocated height to + 2 * EDGE_WIDTH (+ 2)
     * Not only needed by the (removed) MC code but for the iDWT too */
    f->width  = avctx->width  + 2 * EDGE_WIDTH;
    f->height = avctx->height + 2 * EDGE_WIDTH + 2;
    ret = ff_get_buffer(avctx, f, flags);
    if (ret < 0)
        return ret;

    f->width  = avctx->width;
    f->height = avctx->height;

    return 0;
}

/**
 * Dirac Specification ->
 * 11.1.1 Picture Header. picture_header()
 */
static int dirac_decode_picture_header(DiracContext *s)
{
    int ret;

    /* [DIRAC_STD] 11.3 Wavelet transform data */
    if ((ret = dirac_unpack_idwt_params(s)) < 0)
        return ret;

    /* Allocates the idwt coefficient buffers */
    init_planes(s);
    return 0;
}

/**
 * Dirac Specification ->
 * 9.6 Parse Info Header Syntax. parse_info()
 * 4 byte start code + byte parse code + 4 byte size + 4 byte previous size
 */
#define DATA_UNIT_HEADER_SIZE 13

/* [DIRAC_STD] dirac_decode_data_unit makes reference to the while defined in 9.3
   inside the function parse_sequence() */
static int dirac_decode_data_unit(AVCodecContext *avctx,
                                  const uint8_t *buf, int size)
{
    int ret;
    AVFrame *pic;
    uint32_t pict_num;
    uint8_t parse_code;
    int num_slices = -1;
    AVDiracSeqHeader *dsh;
    DiracContext *s = avctx->priv_data;

    /* Must be corruption since the packet isn't large enough to fill a header */
    if (size < DATA_UNIT_HEADER_SIZE)
        return AVERROR_INVALIDDATA;

    /* Get the parse code identifier */
    parse_code = buf[4];

    /* Skip the packet-like header and go directly to the data */
    init_get_bits(&s->gb, &buf[13], 8*(size - DATA_UNIT_HEADER_SIZE));

    /* Sequence header - not actually always required, some encoders may put
     * one sequence header every lots of frames */
    if (parse_code == DIRAC_PCODE_SEQ_HEADER) {
        if (s->seen_sequence_header)
            return 0;

        /* [DIRAC_STD] 10. Sequence header */
        ret = av_dirac_parse_sequence_header(&dsh, buf + DATA_UNIT_HEADER_SIZE, size - DATA_UNIT_HEADER_SIZE, avctx);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "error parsing sequence header");
            return ret;
        }

        ret = ff_set_dimensions(avctx, dsh->width, dsh->height);

        /* In case of interlacing halve the internal height */
        if (dsh->field_coding)
            dsh->height >>= 1;

        /* Error during decoding the sequence header */
        if (ret < 0) {
            av_freep(&dsh);
            return ret;
        }

        /* Error on on change in pixel formats in the second field */
        if (s->cur_field && dsh->pix_fmt != s->prev_field_fmt) {
            av_log(avctx, AV_LOG_ERROR, "Second field has different pixel format!\n");
            av_freep(&dsh);
            return AVERROR_INVALIDDATA;
        } else {
            s->prev_field_fmt = dsh->pix_fmt;
        }

        ff_set_sar(avctx, dsh->sample_aspect_ratio);
        avctx->pix_fmt         = dsh->pix_fmt;
        avctx->color_range     = dsh->color_range;
        avctx->color_trc       = dsh->color_trc;
        avctx->color_primaries = dsh->color_primaries;
        avctx->colorspace      = dsh->colorspace;
        avctx->profile         = dsh->profile;
        avctx->level           = dsh->level;
        avctx->framerate       = dsh->framerate;
        avctx->field_order     = dsh->top_field_first ? AV_FIELD_TT : avctx->field_order;
        s->field_coding        = dsh->field_coding;
        s->bit_depth           = dsh->bit_depth;
        s->version.major       = dsh->version.major;
        s->version.minor       = dsh->version.minor;
        s->seq                 = *dsh;
        av_freep(&dsh);

        s->is_fragment = 0;
        s->fragment_slice_count = 0;
        s->fragment_slices_received = 0;

        s->pshift = s->bit_depth > 8;

        avcodec_get_chroma_sub_sample(avctx->pix_fmt, &s->chroma_x_shift, &s->chroma_y_shift);

        if (avctx->width   != s->seq_buf_allocated_width ||
            avctx->width   != s->seq_buf_allocated_width ||
            avctx->pix_fmt != s->seq_buf_allocated_fmt   ||
            s->field_coding != s->seq_buf_allocated_coding) {
            free_sequence_buffers(s);
            s->seq_buf_allocated_width  = avctx->width;
            s->seq_buf_allocated_height = avctx->height;
            s->seq_buf_allocated_fmt    = avctx->pix_fmt;
            s->seq_buf_allocated_coding = s->field_coding;
            ret = alloc_sequence_buffers(s);
            if (ret < 0)
                return ret;
        }

        s->seen_sequence_header = 1;
    } else if (parse_code == DIRAC_PCODE_END_SEQ) { /* [DIRAC_STD] End of Sequence */
        s->seen_sequence_header = 0;
    } else if (parse_code == DIRAC_PCODE_AUX) {
        ; /* Usually contains encoder information */
    } else if ((parse_code & 0x88) == 0x88) {  /* picture data unit */
        if (!s->seen_sequence_header) {
            av_log(avctx, AV_LOG_DEBUG, "Dropping frame without sequence header\n");
            return AVERROR_INVALIDDATA;
        }

        /* Check that the picture parse code is supported */
        if (!((parse_code & 0xF8) == 0xE8)) {
            av_log(avctx, AV_LOG_ERROR, "Only the VC2 HQ profile is supported!\n");
            return AVERROR_INVALIDDATA;
        }

        s->is_fragment = (parse_code & 0x0C) == 0x0C;

        pic = s->dummy_frame; // TODO: use something else when doing interlaced.
        pic->key_frame = 1;
        pic->pict_type = AV_PICTURE_TYPE_I;

        align_get_bits(&s->gb);

        /* [DIRAC_STD] 11.1.1 Picture Header. picture_header(), also 14.2 for fragment */
        pict_num = get_bits_long(&s->gb, 32);
        av_log(s->avctx, AV_LOG_DEBUG, "Picture number: %d\n", pict_num);

        /* Read the fragment header of the first fragment */
        if (s->is_fragment) {
            unsigned temp = show_bits_long(&s->gb, 32);
            s->fragment_data_length = temp >> 16;
            num_slices = s->fragment_slice_count = temp & 0xFFFF;

            //s->fragment_data_length = get_bits_long(&s->gb, 16); /* fragment_data_length */
            //s->fragment_slice_count = get_bits_long(&s->gb, 16);
            /* TODO: maybe read x and y offset here */
        }

        if (!s->is_fragment || (s->is_fragment && s->fragment_slice_count == 0)) {
            if (!s->field_coding) {
                if ((ret = get_buffer_with_edge(avctx, pic, 0)) < 0)
                    return ret;
                pic->display_picture_number = pict_num;
                s->current_picture = pic;
                s->plane[0].stride = pic->linesize[0];
                s->plane[1].stride = pic->linesize[1];
                s->plane[2].stride = pic->linesize[2];
                s->cur_field = 0;
            } else {
                /* Picks the field number based on the parity of the picture number */
                s->cur_field = pict_num & 1;

                if (!s->cur_field) {
                    av_frame_unref(s->prev_field);
                    if ((ret = get_buffer_with_edge(avctx, pic, AV_GET_BUFFER_FLAG_REF)) < 0)
                        return ret;
                    s->prev_field = pic;
                    s->plane[0].stride = pic->linesize[0];
                    s->plane[1].stride = pic->linesize[1];
                    s->plane[2].stride = pic->linesize[2];
                } else {
                    if (!s->current_picture || !s->prev_field)
                        return AVERROR_INVALIDDATA;
                    /* Check and reject second field having different dimensions */
                    if ((avctx->width  != s->current_picture->width ) ||
                        (avctx->height != s->current_picture->height)) {
                        av_log(avctx, AV_LOG_ERROR, "Second field has different width/height!\n");
                        return AVERROR_INVALIDDATA;
                    }
                    s->plane[0].stride = s->current_picture->linesize[0];
                    s->plane[1].stride = s->current_picture->linesize[1];
                    s->plane[2].stride = s->current_picture->linesize[2];
                }

                s->prev_field->display_picture_number = pict_num;
                s->current_picture = s->prev_field;
            }
        }

        if (!s->is_fragment || (s->is_fragment && s->fragment_slice_count == 0)) {
            /* because we didn't _read_ the fragment_data_length and
             * fragment_slice_count above we need to skip those 32-bits here. */
            if (s->is_fragment)
                skip_bits(&s->gb, 32);

            /* [DIRAC_STD] 11.1 Picture parse. picture_parse() */
            if ((ret = dirac_decode_picture_header(s))) {
                return ret;
            }

            /* Spec 14.3 part of initialize_fragment_state */
            s->fragment_slices_received = 0;
        }

        if (!s->is_fragment || (s->is_fragment && s->fragment_slice_count == 0)) {
            /* Will warn if the encoder's not fast enough or the decoder's not fast
            * enough or if a frame wasn't able to be decoded and was dropped */
            if (s->current_picture->display_picture_number &&
                s->current_picture->display_picture_number != s->prev_pict_number + 1)
                av_log(s->avctx, AV_LOG_WARNING,
                    "Picture number is not linearly incrementing, %i -> %i\n",
                    s->prev_pict_number, s->current_picture->display_picture_number);

            s->prev_pict_number = s->current_picture->display_picture_number;
        }

        if (s->is_fragment && !num_slices)
            return 0;

        /* [DIRAC_STD] 13.0 Transform data syntax. transform_data() */
        ret = dirac_decode_frame_internal(s);
        if (ret < 0)
            return ret;
    }
    return 0;
}

static int dirac_decode_frame(AVCodecContext *avctx, void *data, int *got_frame, AVPacket *pkt)
{
    DiracContext *s     = avctx->priv_data;
    uint8_t *buf        = pkt->data;
    int buf_size        = pkt->size;
    int buf_idx         = 0;
    int ret, picture_element_present = 0;
    unsigned data_unit_size;
    AVFrame *f = data;

    *got_frame = 0;

    /* The parser screwed up - this can happen */
    if (buf_size == 13)
        return 0;

    /* Loop around the buffer until it finds a parse code since the standard
     * structure of a Sequence header, then an Aux header (encoder info), then
     * the actual frame/field and finally the End of sequence header is not
     * required and can be out of order with some parts even missing (seq header)!
     * Though we reject if the sequence header is after the picture header */
    for (;;) {
        /*[DIRAC_STD] Here starts the code from parse_info() defined in 9.6
          [DIRAC_STD] PARSE_INFO_PREFIX = "BBCD" as defined in ISO/IEC 646
          BBCD start code search */
        for (; buf_idx + DATA_UNIT_HEADER_SIZE < buf_size; buf_idx++) {
            if (buf[buf_idx  ] == 'B' && buf[buf_idx+1] == 'B' &&
                buf[buf_idx+2] == 'C' && buf[buf_idx+3] == 'D')
                break;
        }
        /* BBCD found or end of data */
        if (buf_idx + DATA_UNIT_HEADER_SIZE >= buf_size)
            break;

        data_unit_size = AV_RB32(buf+buf_idx+5);
        if (data_unit_size > buf_size - buf_idx || !data_unit_size) {
            if(data_unit_size > buf_size - buf_idx)
            av_log(s->avctx, AV_LOG_ERROR,
                   "Data unit with size %d is larger than input buffer, discarding\n",
                   data_unit_size);
            buf_idx += 4;
            continue;
        }
        /* [DIRAC_STD] dirac_decode_data_unit makes reference to the while defined in 9.3 inside the function parse_sequence() */
        ret = dirac_decode_data_unit(avctx, buf+buf_idx, data_unit_size);
        if (ret < 0) {
            av_log(s->avctx, AV_LOG_ERROR, "Error in dirac_decode_data_unit\n");
            return ret;
        }
        if (s->is_fragment) {
            picture_element_present = s->fragment_slices_received == (s->num_x * s->num_y);
        } else {
            picture_element_present |= *(buf + buf_idx + 4) & 0x8;
        }
        buf_idx += data_unit_size;
    }

    /* ref the top field's frame during field coded interlacing */
    if (s->field_coding) {
        if (!s->current_picture)
            return AVERROR_INVALIDDATA;
        else if ((ret=av_frame_ref(data, s->current_picture)) < 0)
            return ret;
    }

    /* No output for the top field, wait for the second */
    if (s->field_coding && !s->cur_field)
        picture_element_present = 0;

    /* Return a frame only if there was a valid picture in the packet */
    if (picture_element_present) {
        if (s->is_fragment)
            s->avctx->execute2(s->avctx, idwt_plane, NULL, NULL, 3);

        *got_frame = 1;
        // TODO: do something with frame
        //ret = av_frame_ref(data, s->current_picture);
        av_frame_move_ref(data, s->current_picture);
    } else {
        *got_frame = 0;
    }


    /* Total bytes read */
    return buf_idx;
}

AVCodec ff_dirac_decoder = {
    .name           = "dirac",
    .long_name      = NULL_IF_CONFIG_SMALL("BBC Dirac VC-2"),
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_DIRAC,
    .priv_data_size = sizeof(DiracContext),
    .init           = dirac_decode_init,
    .close          = dirac_decode_end,
    .decode         = dirac_decode_frame,
    .capabilities   = AV_CODEC_CAP_SLICE_THREADS | AV_CODEC_CAP_DR1,
    .flush          = dirac_decode_flush,
};
