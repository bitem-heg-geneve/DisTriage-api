# DisTriage API Performance Optimization Report

**Date:** January 9, 2026

## Executive Summary

The DisTriage API was optimized to improve throughput through CPU threading optimization. Benchmark testing with 1000 PMIDs shows processing in ~23 minutes with the optimized configuration.

## Benchmark Results (1000 PMIDs)

| Metric | Value |
|--------|-------|
| Total PMIDs submitted | 1000 |
| Successfully processed | 964 |
| Failed (not found in MEDLINE) | 36 |
| Total time | ~23 minutes |
| **Throughput** | **~42 PMIDs/min** |

*Benchmark performed January 9, 2026 with optimized configuration.*

## Problem Statement

The API worker was experiencing CPU contention due to:
1. No thread limits - each of 4 workers competed for all 16 cores
2. High prefetch multiplier (4) causing bursty load patterns
3. Tokenizer parallelism causing potential deadlocks

## Configuration Changes

### Before (docker-compose.yml)

```yaml
worker:
  environment:
    # No thread limits - workers competed for all 16 cores
  command:
    - "--concurrency=4"
    - "--prefetch-multiplier=4"
```

### After (docker-compose.yml)

```yaml
worker:
  environment:
    OMP_NUM_THREADS: "4"           # 16 cores / 4 workers = 4 threads each
    MKL_NUM_THREADS: "4"           # For Intel MKL operations
    TOKENIZERS_PARALLELISM: "false" # Prevent tokenizer deadlocks
  command:
    - "--concurrency=4"
    - "--prefetch-multiplier=1"    # Reduced from 4 for smoother load
```

## Key Optimizations

### 1. Thread Limiting (OMP_NUM_THREADS=4)
- **Problem:** 4 workers on 16 cores, each trying to use all cores → CPU thrashing
- **Solution:** Limit each worker to 4 threads (16 / 4 = 4)
- **Impact:** Clean CPU allocation, eliminates thread contention

### 2. Reduced Prefetch Multiplier (4 → 1)
- **Problem:** High prefetch causes bursty load, memory pressure
- **Solution:** Prefetch 1 task per worker for smoother processing
- **Impact:** More stable load distribution, reduced memory spikes

### 3. Tokenizer Parallelism Disabled
- Prevents potential deadlocks in forked Celery workers
- Standard practice for production deployments

## Server Specifications

- **CPU:** 16 cores
- **RAM:** 62 GB
- **Model:** BiomedBERT (single fine-tuned checkpoint)
- **Framework:** PyTorch (CPU inference)

## Batch Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| INGRESS_BATCH_SIZE | 64 | PMIDs per SIBiLS fetch batch |
| INFER_BATCH_SIZE | 32 | Documents per inference batch |

## Conclusion

The optimization eliminates CPU contention and provides smoother load distribution. Throughput is ~42 PMIDs/min (~2500/hour), suitable for daily MEDLINE triage batches.
