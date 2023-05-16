#include "ftle.h"

__global__ void gpu_compute_gradient_2D (int nPoints, int offset, int faces_offset, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint,  double *d_logSqrt, double T);
__global__ void gpu_compute_gradient_3D (int nPoints, int offset, int faces_offset, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint,  double *d_logSqrt, double T);
