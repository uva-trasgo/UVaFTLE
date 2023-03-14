#include <stdio.h>
#include <stdlib.h>
#include <CL/sycl.hpp>
using namespace cl::sycl;

#ifndef BLOCK
#define BLOCK 512
#endif
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

void compute_gradient_2D (queue* q,  int nPoints, int offset, int nVertsPerFace, cl::sycl::buffer<double, 1> *b_coords, cl::sycl::buffer<double, 1> *b_flowmap, cl::sycl::buffer<int, 1> *b_faces, 
					cl::sycl::buffer<int, 1> *b_nFacesPerPoint, cl::sycl::buffer<int, 1> *b_facesPerPoint, cl::sycl::buffer<double, 1> *b_log_sqrt, double T );
void compute_gradient_3D (queue* q,  int nPoints, int offset, int nVertsPerFace, cl::sycl::buffer<double, 1> *b_coords, cl::sycl::buffer<double, 1> *b_flowmap, cl::sycl::buffer<int, 1> *b_faces, 
					cl::sycl::buffer<int, 1> *b_nFacesPerPoint, cl::sycl::buffer<int, 1> *b_facesPerPoint, cl::sycl::buffer<double, 1> *b_log_sqrt, double T );
double log_sqrt ( double T, double eigen );
double max_solve_3rd_degree_eq ( double a, double b, double c, double d);
double max_eigen_2D ( double A10, double A11, double A20, double A21 );
double max_eigen_3D ( double A10, double A11, double A12,
                double A20, double A21, double A22,
                double A30, double A31, double A32 );

