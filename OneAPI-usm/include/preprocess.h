#include <stdio.h>
#include "ftle.h"

void read_coordinates ( char *filename, int nDim, int npoints, double *coords );
void read_faces ( char *filename, int nDim, int nVertsPerFace, int nfaces, int *faces );
void read_flowmap ( char *filename, int nDims, int nPoints, double *flowmap );
void create_nFacesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint );
::event create_facesPerPoint_vector (queue* q, int nDim, int nPoints, int offset, int faces_offset, int nFaces, int nVertsPerFace, int* faces, int* nFacesPerPoint, int* facesPerPoint );
