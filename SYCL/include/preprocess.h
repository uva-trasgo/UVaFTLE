#include <stdio.h>
#include <stdlib.h>
#include <CL/sycl.hpp>
using namespace cl::sycl;
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

void read_coordinates ( char *filename, int nDim, int npoints, double *coords );
void read_faces ( char *filename, int nDim, int nVertsPerFace, int nfaces, int *faces );
void read_flowmap ( char *filename, int nDims, int nPoints, double *flowmap );
void create_nFacesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint );
cl::sycl::event create_facesPerPoint_vector (queue* q, int nDim, int nPoints, int offset, int faces_offset, int nFaces, int nVertsPerFace, cl::sycl::buffer<int, 1> *b_faces, cl::sycl::buffer<int, 1> *b_nFacesPerPoint, cl::sycl::buffer<int, 1> *b_facesPerPoint );
