"""Row statistics for dense and paged MQA logits (SM100 only).

`row_stats` is an optional zero-initialized FP32 `[num_rows, 2]` tensor. For every logits
row the kernel accumulates `(sum, sum of squares)` of the stored logits over the row's valid
KV range and leaves the logits themselves untouched. The test checks that the statistics
match an FP64 reduction over the masked kernel logits, that accumulation is additive across
calls, and that passing `row_stats` does not change the logits.
"""
import os
import random
import sys
from typing import Tuple

import torch

import deep_gemm
from deep_gemm.testing import assert_bitwise_equal, get_arch_major
from deep_gemm.utils import (
    ceil_div,
    per_custom_dims_cast_to_fp8, per_token_cast_to_fp8, per_token_cast_to_fp4,
)


def _skip_if_unsupported():
    if get_arch_major() != 10:
        print(f'Skipping row statistics tests on arch major {get_arch_major()}')
        sys.exit(0)


def to_mqa_weights(weights: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    stride = ceil_div(weights.size(1) * element_size, 16) * 16 // element_size
    storage = torch.empty((weights.size(0), stride), device=weights.device, dtype=dtype)
    result = storage[:, :weights.size(1)]
    result.copy_(weights)
    return result


def shard(cases):
    """`DG_ROW_STATS_SHARD=i/n` runs only every n-th case (starting at i), so several processes
    can JIT-compile disjoint kernel variants in parallel."""
    spec = os.getenv('DG_ROW_STATS_SHARD')
    if not spec:
        return cases
    i, n = (int(x) for x in spec.split('/'))
    return [c for k, c in enumerate(cases) if k % n == i]


def ref_row_stats(masked_logits: torch.Tensor) -> torch.Tensor:
    # `masked_logits`: invalid columns already zeroed, FP64 reduction
    x = masked_logits.double()
    return torch.stack([x.sum(dim=1), (x * x).sum(dim=1)], dim=1)


def check_row_stats(name: str, stats: torch.Tensor, ref: torch.Tensor, num_valid: torch.Tensor):
    assert stats.shape == ref.shape, f'{name}: shape {stats.shape} vs {ref.shape}'
    assert torch.isfinite(stats).all(), f'{name}: non-finite statistics'
    # FP32 partials + relaxed FP32 reductions accumulate about eps * sqrt(n) relative error in
    # practice. The sum can cancel, so its error is measured against the L2 scale
    # sqrt(n * sum_sq) (the FP32 error of summing n terms scales with sum |x|), not against |sum|.
    n = num_valid.double().clamp(min=1)
    tol = 1e-5 + 2e-7 * n.sqrt() * 8
    diff = (stats.double() - ref).abs()
    l2_scale = (n * ref[:, 1]).sqrt()
    scale = torch.stack([torch.maximum(ref[:, 0].abs(), l2_scale), ref[:, 1]], dim=1).clamp(min=1e-3)
    rel = diff / scale
    bad = rel > tol.unsqueeze(1)
    if bad.any():
        idx = bad.nonzero()[0]
        raise AssertionError(f'{name}: row {idx[0].item()} stat {idx[1].item()}: '
                             f'kernel {stats[idx[0], idx[1]].item()} ref {ref[idx[0], idx[1]].item()} '
                             f'rel {rel[idx[0], idx[1]].item():.3e} tol {tol[idx[0]].item():.3e}')
    return rel.max().item()


def test_dense_row_stats():
    print('Testing dense MQA logits row statistics:')

    def generate_ks_ke(seq_len: int, seq_len_kv: int, disable_cp: bool):
        if disable_cp:
            ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
            ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)
            return ks, ke
        chunk_size = seq_len // 2
        cp_size = seq_len_kv // seq_len
        cp_id = cp_size // 3
        ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
        ke = torch.zeros(seq_len, dtype=torch.int, device='cuda')
        idx = torch.arange(chunk_size, device='cuda', dtype=torch.int)
        ke[:chunk_size] = cp_id * chunk_size + idx
        ke[chunk_size:] = (cp_size * 2 - 1 - cp_id) * chunk_size + idx
        return ks, ke

    def enumerate_cases():
        for fmt in ('fp8', 'mxfp8', 'mxfp4'):
            for logits_dtype, weights_dtype in ((torch.float, torch.float), (torch.bfloat16, torch.float), (torch.bfloat16, torch.bfloat16)):
                for compressed, clean in ((True, False), (False, False)):
                    for seq_len, seq_len_kv in ((1000, 8192), (2048, 8192), (4096, 65536)):
                        for num_heads in (8, 12, 16, 32, 64):
                            for head_dim in ((64, 128) if fmt == 'mxfp4' else (32, 64, 128)):
                                for disable_cp in (False, True):
                                    if not disable_cp and (seq_len_kv % seq_len != 0 or seq_len % 2 != 0):
                                        continue
                                    yield fmt, logits_dtype, weights_dtype, compressed, clean, seq_len, seq_len_kv, num_heads, head_dim, disable_cp

    cases = list(enumerate_cases())
    num_cases = int(os.getenv('DG_ROW_STATS_NUM_CASES', '40'))
    rng = random.Random(1234)
    selected = rng.sample(cases, min(num_cases, len(cases)))
    # Always include the DeepSeek-style H=64/D=128 configs
    selected += [c for c in cases if c[7] == 64 and c[8] == 128 and c[5] == 2048][:12]
    selected = shard(selected)
    print(f' > running {len(selected)}/{len(cases)} dense cases')

    for fmt, logits_dtype, weights_dtype, compressed, clean, seq_len, seq_len_kv, num_heads, head_dim, disable_cp in selected:
        q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
        kernel_weights = to_mqa_weights(weights, weights_dtype)
        ks, ke = generate_ks_ke(seq_len, seq_len_kv, disable_cp)

        if fmt in ('mxfp4', 'mxfp8'):
            cast_fwd = per_token_cast_to_fp4 if fmt == 'mxfp4' else per_token_cast_to_fp8
            elem_dim = head_dim // 2 if fmt == 'mxfp4' else head_dim
            q_q = cast_fwd(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            q_in = (q_q[0].view(seq_len, num_heads, elem_dim), q_q[1].view(seq_len, num_heads))
            kv_q = cast_fwd(kv.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            kv_in = (kv_q[0].view(seq_len_kv, elem_dim), kv_q[1].view(seq_len_kv))
        else:
            q_in = q.to(torch.float8_e4m3fn), None
            kv_in = per_custom_dims_cast_to_fp8(kv, (0, ), False)

        kwargs = dict(q=q_in, kv=kv_in, weights=kernel_weights, cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                      clean_logits=clean, max_seqlen_k=0, logits_dtype=logits_dtype)
        if compressed:
            kwargs['max_seqlen_k'] = (ke - ks).max().item()

        # Reference logits without statistics
        ref_logits = deep_gemm.fp8_fp4_mqa_logits(**kwargs)
        cols = torch.arange(ref_logits.size(1), device='cuda')
        if compressed:
            valid = cols[None, :] < (ke - ks)[:, None]
        else:
            valid = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
        ref_masked = ref_logits.float().masked_fill(~valid, 0)
        ref_stats = ref_row_stats(ref_masked)
        num_valid = valid.sum(dim=1)

        workspace = deep_gemm.get_mqa_logits_metadata(ks, ke, seq_len_kv, num_heads)
        for use_schedule in (False, True):
            # Alternate between the compact `[rows, 2]` layout and a 32-byte padded row layout
            stats = torch.zeros(seq_len, 2 if use_schedule else 8, device='cuda', dtype=torch.float32)[:, :2]
            extra = dict(schedule_meta=workspace) if use_schedule else dict()
            logits = deep_gemm.fp8_fp4_mqa_logits(**kwargs, row_stats=stats, **extra)
            name = f'dense fmt={fmt} L={logits_dtype} W={weights_dtype} CMP={int(compressed)} CLEAN={int(clean)} SQ={seq_len} SK={seq_len_kv} H={num_heads} D={head_dim} CP={int(not disable_cp)} sched={int(use_schedule)}'
            assert_bitwise_equal(logits.masked_fill(~valid, 0), ref_masked.to(logits.dtype), name + ' (logits unchanged)')
            rel = check_row_stats(name, stats, ref_stats, num_valid)
            # Accumulating twice must double the statistics (zero-init contract)
            deep_gemm.fp8_fp4_mqa_logits(**kwargs, row_stats=stats, **extra)
            check_row_stats(name + ' (x2)', stats, ref_stats * 2, num_valid)
            print(f' > {name}: max rel err {rel:.2e}')
        del ref_logits, ref_masked, valid
    print()


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[:, :block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def kv_cache_cast_to_mx(x: torch.Tensor, is_fp4: bool) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    cast_fwd = per_token_cast_to_fp4 if is_fp4 else per_token_cast_to_fp8
    x_scaled, sf = cast_fwd(x.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    elem = head_dim // 2 if is_fp4 else head_dim
    x_mx = torch.empty((num_blocks, block_size * (elem + 4)), device=x.device, dtype=torch.uint8)
    x_mx[:, :block_size * elem] = x_scaled.view(num_blocks, block_size * elem).view(torch.uint8)
    x_mx[:, block_size * elem:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_mx.view(num_blocks, block_size, num_heads, elem + 4)


def test_paged_row_stats():
    print('Testing paged MQA logits row statistics:')

    def enumerate_cases():
        for is_varlen in (False, True):
            for fmt in ('fp8', 'mxfp8', 'mxfp4'):
                for logits_dtype, weights_dtype in ((torch.float, torch.float), (torch.bfloat16, torch.float), (torch.bfloat16, torch.bfloat16)):
                    for block_kv in (32, 64, 128):
                        for batch_size in (17, 256):
                            for next_n in ((1, ) if is_varlen else (1, 2, 6)):
                                for max_tokens_per_seq in ((3, 7) if is_varlen else (1, )):
                                    for num_heads in (8, 16, 32, 64):
                                        for head_dim in ((64, 128) if fmt == 'mxfp4' else (32, 128)):
                                            for avg_kv in (1000, 8192, 65536):
                                                if batch_size * avg_kv > 8 * 1024 * 1024:
                                                    continue
                                                yield is_varlen, fmt, logits_dtype, weights_dtype, block_kv, batch_size, next_n, max_tokens_per_seq, num_heads, head_dim, avg_kv

    cases = list(enumerate_cases())
    num_cases = int(os.getenv('DG_ROW_STATS_NUM_CASES', '40'))
    rng = random.Random(4321)
    selected = rng.sample(cases, min(num_cases, len(cases)))
    selected += [c for c in cases if c[8] == 64 and c[9] == 128 and c[1] == 'fp8' and c[4] == 64][:12]
    selected = shard(selected)
    print(f' > running {len(selected)}/{len(cases)} paged cases')

    for is_varlen, fmt, logits_dtype, weights_dtype, block_kv, batch_size, next_n, max_tokens_per_seq, num_heads, head_dim, avg_kv in selected:
        raw_batch_size = batch_size
        if is_varlen:
            tokens_per_seq = torch.randint(1, max_tokens_per_seq + 1, (raw_batch_size,), device='cuda', dtype=torch.int)
            indices = torch.arange(raw_batch_size, device='cuda', dtype=torch.int).repeat_interleave(tokens_per_seq)
            batch_size, next_n = tokens_per_seq.sum().item(), 1
        else:
            tokens_per_seq, indices = None, None

        q = torch.randn((batch_size, next_n, num_heads, head_dim), device='cuda', dtype=torch.bfloat16)
        weights = torch.randn((batch_size * next_n, num_heads), device='cuda', dtype=torch.float)
        kernel_weights = to_mqa_weights(weights, weights_dtype)
        context_lens = torch.randint(max(1, int(0.7 * avg_kv)), int(1.3 * avg_kv) + 1, (raw_batch_size,), device='cuda', dtype=torch.int)
        max_ctx_len_per_seq = context_lens + (tokens_per_seq - 1) if is_varlen else context_lens

        num_blocks_per_query = ceil_div(max_ctx_len_per_seq, block_kv)
        max_model_len = num_blocks_per_query.max().item() * block_kv
        num_total_blocks = num_blocks_per_query.sum().item()
        kv_cache = torch.randn((num_total_blocks, block_kv, 1, head_dim), device='cuda', dtype=torch.bfloat16)
        block_table = torch.zeros((raw_batch_size, num_blocks_per_query.max().item()), device='cuda', dtype=torch.int)
        block_idx_pool = torch.randperm(num_total_blocks, device='cuda', dtype=torch.int)
        offset = 0
        for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
            block_table[i, :num_blocks] = block_idx_pool[offset:offset + num_blocks]
            offset += num_blocks
        if is_varlen:
            context_lens = context_lens.repeat_interleave(tokens_per_seq)
            offsets_within_seq = torch.cat([torch.arange(n.item(), device='cuda', dtype=torch.int) for n in tokens_per_seq])
            context_lens = context_lens + offsets_within_seq
            block_table = block_table.repeat_interleave(tokens_per_seq, dim=0)

        if fmt in ('mxfp4', 'mxfp8'):
            cast_fwd = per_token_cast_to_fp4 if fmt == 'mxfp4' else per_token_cast_to_fp8
            elem_dim = head_dim // 2 if fmt == 'mxfp4' else head_dim
            q_q = cast_fwd(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            q_in = (q_q[0].view(batch_size, next_n, num_heads, elem_dim), q_q[1].view(batch_size, next_n, num_heads))
            kv_in = kv_cache_cast_to_mx(kv_cache, fmt == 'mxfp4')
        else:
            q_in = q.to(torch.float8_e4m3fn), None
            kv_in = kv_cache_cast_to_fp8(kv_cache)
        del q, kv_cache

        # 2D context lens: one valid length per token row
        if is_varlen:
            context_lens_nextn = context_lens.view(-1, 1)
        else:
            context_lens_nextn = ((context_lens.unsqueeze(1) + 1) * torch.rand(batch_size, next_n, device='cuda')).int().clamp(min=1)
            context_lens_nextn[:, -1] = context_lens
        row_lens = context_lens_nextn.view(-1)
        positions = torch.arange(max_model_len, device='cuda').unsqueeze(0)
        valid = positions < row_lens[:, None]

        metadata_kwargs = dict(context_lens=context_lens_nextn, block_kv=block_kv, num_sms=deep_gemm.get_num_sms(), indices=indices)
        kwargs = dict(q=q_in, kv_cache=kv_in, weights=kernel_weights, context_lens=context_lens_nextn, block_table=block_table,
                      schedule_meta=deep_gemm.get_paged_mqa_logits_metadata(**metadata_kwargs),
                      max_context_len=max_model_len, clean_logits=False, logits_dtype=logits_dtype, indices=indices)
        ref_logits = deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs)
        ref_masked = ref_logits.float().masked_fill(~valid, 0)
        ref_stats = ref_row_stats(ref_masked)
        num_valid = valid.sum(dim=1)

        stats = torch.zeros(batch_size * next_n, 8 if (batch_size % 2 == 0) else 2, device='cuda', dtype=torch.float32)[:, :2]
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs, row_stats=stats)
        name = f'paged fmt={fmt} L={logits_dtype} W={weights_dtype} VAR={int(is_varlen)} PAGE={block_kv} BSZ={raw_batch_size} NextN={next_n} H={num_heads} D={head_dim} L={avg_kv}'
        assert_bitwise_equal(logits.masked_fill(~valid, 0), ref_masked.to(logits.dtype), name + ' (logits unchanged)')
        rel = check_row_stats(name, stats, ref_stats, num_valid)
        deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs, row_stats=stats)
        check_row_stats(name + ' (x2)', stats, ref_stats * 2, num_valid)
        print(f' > {name}: max rel err {rel:.2e}')
        del ref_logits, ref_masked, valid, kwargs, q_in, kv_in
        torch.cuda.empty_cache()
    print()


def test_legacy_wrappers():
    # `fp8_mqa_logits` / `fp8_paged_mqa_logits` accept `row_stats` as well
    print('Testing legacy wrappers with row statistics:')
    seq_len, seq_len_kv, num_heads, head_dim = 512, 4096, 64, 128
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    kv_in = per_custom_dims_cast_to_fp8(kv, (0, ), False)
    weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)
    max_seqlen_k = (ke - ks).max().item()
    stats = torch.zeros(seq_len, 2, device='cuda', dtype=torch.float32)
    logits = deep_gemm.fp8_mqa_logits(q, kv_in, weights, ks, ke, clean_logits=False, max_seqlen_k=max_seqlen_k, row_stats=stats)
    valid_mask = torch.arange(logits.size(1), device='cuda')[None, :] < (ke - ks)[:, None]
    masked = logits.masked_fill(~valid_mask, 0)
    rel = check_row_stats('legacy dense', stats, ref_row_stats(masked), valid_mask.sum(dim=1))
    print(f' > legacy dense: max rel err {rel:.2e}')

    # Wrong shapes / dtypes / layouts and cleaning must be rejected
    bad_cases = [dict(row_stats=torch.zeros(seq_len, 3, device='cuda')[:, :2]),                       # 12-byte rows: misaligned pairs
                 dict(row_stats=torch.zeros(seq_len, 2, device='cuda', dtype=torch.bfloat16)),
                 dict(row_stats=torch.zeros(seq_len + 1, 2, device='cuda')),
                 dict(row_stats=torch.zeros(seq_len, 4, device='cuda')[:, ::2]),                      # non-unit inner stride
                 dict(row_stats=torch.zeros(seq_len, 2, device='cuda'), clean_logits=True, max_seqlen_k=0)]
    for bad in bad_cases:
        kwargs = dict(clean_logits=False, max_seqlen_k=max_seqlen_k)
        kwargs.update(bad)
        try:
            deep_gemm.fp8_mqa_logits(q, kv_in, weights, ks, ke, **kwargs)
        except Exception:
            continue
        raise AssertionError(f'Invalid row_stats call accepted: {bad}')
    print(' > invalid row_stats tensors / cleaning combination rejected')
    print()


if __name__ == '__main__':
    _skip_if_unsupported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    random.seed(0)

    test_legacy_wrappers()
    test_dense_row_stats()
    test_paged_row_stats()
    print('All row statistics tests passed')
