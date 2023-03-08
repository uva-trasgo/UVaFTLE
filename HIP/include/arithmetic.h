#ifndef STRUCTS_H
#define STRUCTS_H
typedef struct Face {
   int      index;      /* index to mesh->faces structure */
   int     *vertices;   /* index to mesh->points structure */
} face_t;

typedef struct Point {
   int       index; // Point index in mesh->points
   double   *coordinates;
   int       nfaces;
   int      *faces;
} point_t;

typedef struct Mesh {
   int       nDim;
   int       nPoints;
   int       nFaces;
   int       nVertsPerFace;
   point_t  *points;
   face_t   *faces;
} mesh_t;
#endif

__global__ void gpu_compute_gradient_2D (int stride, int numCoords, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *ftl_matrix, double* d_W_ei, double *d_logSqrt, double T);
__global__ void gpu_compute_gradient_3D (int stride, int numCoords, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *ftl_matrix, double* d_W_ei, double *d_logSqrt, double T);
__global__ void gpu_log_sqrt_2D(int numCoords, double T, double * d_w, double * d_r );
__global__ void gpu_log_sqrt_3D(int numCoords, double T, double * d_w, double * d_r);
