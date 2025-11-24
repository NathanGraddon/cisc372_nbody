#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

extern vector3 *d_hPos, *d_hVel;
extern double *d_mass;

extern vector3 *hPos, *hVel;
extern double  *mass;

// Kernel 1: compute pairwise accelerations accels[i*n + j]
__global__
void compute_pairwise_accels(const vector3 *pos,
                             const double  *m,
                             vector3 *accels,
                             int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    int idx = i * n + j;

    if (i == j) {
        accels[idx][0] = 0.0;
        accels[idx][1] = 0.0;
        accels[idx][2] = 0.0;
        return;
    }

    double dx = pos[i][0] - pos[j][0];
    double dy = pos[i][1] - pos[j][1];
    double dz = pos[i][2] - pos[j][2];

    double mag_sq = dx*dx + dy*dy + dz*dz;
    double mag    = sqrt(mag_sq);

    double accelmag = -GRAV_CONSTANT * m[j] / mag_sq;

    accels[idx][0] = accelmag * dx / mag;
    accels[idx][1] = accelmag * dy / mag;
    accels[idx][2] = accelmag * dz / mag;
}

// Kernel 2: sum row i and update vel/pos for object i
__global__
void sum_and_update(const vector3 *accels,
                    vector3 *pos,
                    vector3 *vel,
                    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ax = 0.0, ay = 0.0, az = 0.0;

    int base = i * n;
    for (int j = 0; j < n; j++) {
        ax += accels[base + j][0];
        ay += accels[base + j][1];
        az += accels[base + j][2];
    }

    // v = v + a*dt
    vel[i][0] += ax * INTERVAL;
    vel[i][1] += ay * INTERVAL;
    vel[i][2] += az * INTERVAL;

    // x = x + v*dt
    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

static void compute_gpu_cuda()
{
    static vector3 *d_accels = NULL;
    static int alloc_n = 0;

    if (d_accels == NULL || alloc_n != NUMENTITIES) {
        if (d_accels) cudaFree(d_accels);
        cudaMalloc((void**)&d_accels,
                   sizeof(vector3) * NUMENTITIES * NUMENTITIES);
        alloc_n = NUMENTITIES;
    }

    dim3 block2d(16, 16);
    dim3 grid2d((NUMENTITIES + block2d.x - 1)/block2d.x,
                (NUMENTITIES + block2d.y - 1)/block2d.y);

    compute_pairwise_accels<<<grid2d, block2d>>>(
        d_hPos, d_mass, d_accels, NUMENTITIES
    );

    int block1d = 256;
    int grid1d  = (NUMENTITIES + block1d - 1)/block1d;

    sum_and_update<<<grid1d, block1d>>>(
        d_accels, d_hPos, d_hVel, NUMENTITIES
    );

    cudaDeviceSynchronize();
}

static void compute_cpu_omp()
{
    int i, j, k;

    vector3 *values = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    vector3 **accels = (vector3**)malloc(sizeof(vector3*) * NUMENTITIES);
    for (i = 0; i < NUMENTITIES; i++)
        accels[i] = &values[i * NUMENTITIES];

    // OpenMP kernel #1: parallelize pairwise accelerations (n^2 work)
    #pragma omp parallel for collapse(2) private(k)
    for (i = 0; i < NUMENTITIES; i++) {
        for (j = 0; j < NUMENTITIES; j++) {

            if (i == j) {
                FILL_VECTOR(accels[i][j], 0, 0, 0);
            } else {
                vector3 distance;
                for (k = 0; k < 3; k++)
                    distance[k] = hPos[i][k] - hPos[j][k];

                double mag_sq = distance[0]*distance[0]
                              + distance[1]*distance[1]
                              + distance[2]*distance[2];
                double mag = sqrt(mag_sq);

                double accelmag = -GRAV_CONSTANT * mass[j] / mag_sq;

                FILL_VECTOR(accels[i][j],
                            accelmag * distance[0] / mag,
                            accelmag * distance[1] / mag,
                            accelmag * distance[2] / mag);
            }
        }
    }

    // OpenMP kernel #2: parallelize sum+update per body (n bodies)
    #pragma omp parallel for private(j, k)
    for (i = 0; i < NUMENTITIES; i++) {
        vector3 accel_sum = {0, 0, 0};

        for (j = 0; j < NUMENTITIES; j++) {
            for (k = 0; k < 3; k++)
                accel_sum[k] += accels[i][j][k];
        }

        for (k = 0; k < 3; k++) {
            hVel[i][k] += accel_sum[k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
    }

    free(accels);
    free(values);
}


// Default: CUDA path (so current nbody.c works).
void compute()
{
#ifdef USE_OMP_CPU
    compute_cpu_omp();
#else
    compute_gpu_cuda();
#endif
}