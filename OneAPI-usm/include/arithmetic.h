#include "ftle.h"

::event compute_gradient_2D (::event* dep_event, queue* q,  int nPoints, int offset, int faces_offset, int nVertsPerFace, double* coords, double* flowmap, int* faces, int* nFacesPerPoint, int* facesPerPoint, double* logSqrt, double T );
::event compute_gradient_3D (::event* dep_event, queue* q,  int nPoints, int offset, int faces_offset, int nVertsPerFace, double* coords, double* flowmap, int* faces, int* nFacesPerPoint, int* facesPerPoint, double* logSqrt, double T );
