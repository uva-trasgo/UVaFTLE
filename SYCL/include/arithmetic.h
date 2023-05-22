#include "ftle.h"

::event compute_gradient_2D (queue* q,  int nPoints, int offset, int faces_offset, int nVertsPerFace, ::buffer<double, 1> *b_coords, ::buffer<double, 1> *b_flowmap, ::buffer<int, 1> *b_faces, ::buffer<int, 1> *b_nFacesPerPoint, ::buffer<int, 1> *b_facesPerPoint, ::buffer<double, 1> *b_log_sqrt, double T );
::event compute_gradient_3D (queue* q,  int nPoints, int offset, int faces_offset, int nVertsPerFace, ::buffer<double, 1> *b_coords, ::buffer<double, 1> *b_flowmap, ::buffer<int, 1> *b_faces, ::buffer<int, 1> *b_nFacesPerPoint, ::buffer<int, 1> *b_facesPerPoint, ::buffer<double, 1> *b_log_sqrt, double T );
