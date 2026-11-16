/* stream_mini.c - tiny memory-bandwidth + in-cache FP throughput benchmark.
 * Build: gcc -O3 -march=native -fopenmp stream_mini.c -o stream_mini
 * Usage: ./stream_mini [array_megabytes_per_array (default 320)]
 * Output: one "key=value" per line so the Python driver can parse it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 320;
    size_t n = mb * 1024 * 1024 / sizeof(double);
    double *a, *b, *c;
    if (posix_memalign((void **)&a, 64, n * sizeof(double)) ||
        posix_memalign((void **)&b, 64, n * sizeof(double)) ||
        posix_memalign((void **)&c, 64, n * sizeof(double))) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("threads=%d\narray_mb=%zu\n", threads, mb);

    /* parallel first touch so pages land on the NUMA node of the thread using them */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    double best_copy = 1e30, best_triad = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        double t = now();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) c[i] = a[i];
        t = now() - t;
        if (t < best_copy) best_copy = t;

        t = now();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
        t = now() - t;
        if (t < best_triad) best_triad = t;
    }
    double bytes = (double)n * sizeof(double);
    printf("copy_gbps=%.2f\n", 2.0 * bytes / best_copy / 1e9);
    printf("triad_gbps=%.2f\n", 3.0 * bytes / best_triad / 1e9);

    /* in-cache daxpy: each thread works on its own 16 KiB arrays (fits L1),
     * so this measures vectorized FP throughput, not memory. 2 flops/element. */
    enum { M = 1024 };
    long reps = 400000;
    double sum = 0.0;
    double t = now();
    #pragma omp parallel reduction(+:sum)
    {
        double x[M] __attribute__((aligned(64)));
        double y[M] __attribute__((aligned(64)));
        for (int i = 0; i < M; i++) { x[i] = i * 1e-3; y[i] = 1.0; }
        for (long r = 0; r < reps; r++) {
            double alpha = 1.0 + 1e-9 * (double)(r & 7);
            #pragma omp simd
            for (int i = 0; i < M; i++) y[i] = alpha * x[i] + y[i] * 0.999999;
        }
        sum += y[M / 2];
    }
    t = now() - t;
    /* 3 flops per element (mul, mul, add) */
    printf("incache_gflops=%.2f\n", 3.0 * M * (double)reps * threads / t / 1e9);
    printf("checksum=%.6g\n", sum + a[n / 2]);
    free(a); free(b); free(c);
    return 0;
}
