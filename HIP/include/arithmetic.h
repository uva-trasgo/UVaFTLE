#include "ftle.h"

__global__ void gpu_compute_gradient_2D (int stride, int numCoords, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint,  double *d_logSqrt, double T);
__global__ void gpu_compute_gradient_3D (int stride, int numCoords, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint,  double *d_logSqrt, double T);
__global__ void gpu_log_sqrt_2D(int numCoords, double T, double * d_w, double * d_r );
__global__ void gpu_log_sqrt_3D(int numCoords, double T, double * d_w, double * d_r);
