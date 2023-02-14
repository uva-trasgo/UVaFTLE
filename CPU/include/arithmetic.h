#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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

void compute_gradient_2D ( int ip, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *log_sqrt, double T );
void compute_gradient_3D ( int ip, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *log_sqrt, double T );
double log_sqrt ( double T, double eigen );
double max_solve_3rd_degree_eq ( double a, double b, double c, double d);
double max_eigen_2D ( double A10, double A11, double A20, double A21 );
double max_eigen_3D ( double A10, double A11, double A12,
                double A20, double A21, double A22,
                double A30, double A31, double A32 );

