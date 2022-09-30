#include <stdio.h>
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

void read_coordinates ( char *filename, int nDim, int npoints, double *coords );
void read_faces ( char *filename, int nDim, int nVertsPerFace, int nfaces, int *faces );
void read_flowmap ( char *filename, int nDims, int nPoints, double *flowmap );
void create_nFacesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint );
void create_facesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint, int *facesPerPoint );
__global__ void create_facesPerPoint_vector_GPU ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint, int *facesPerPoint );
