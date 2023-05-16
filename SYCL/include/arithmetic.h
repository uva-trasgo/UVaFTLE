#include "ftle.h"

::event compute_gradient_2D (queue* q,  int nPoints, int offset, int faces_offset, int nVertsPerFace, ::buffer<double, 1> *b_coords, ::buffer<double, 1> *b_flowmap, ::buffer<int, 1> *b_faces, ::buffer<int, 1> *b_nFacesPerPoint, ::buffer<int, 1> *b_facesPerPoint, ::buffer<double, 1> *b_log_sqrt, double T );
::event compute_gradient_3D (queue* q,  int nPoints, int offset, int faces_offset, int nVertsPerFace, int faces_offset,::buffer<double, 1> *b_coords, ::buffer<double, 1> *b_flowmap, ::buffer<int, 1> *b_faces, ::buffer<int, 1> *b_nFacesPerPoint, ::buffer<int, 1> *b_facesPerPoint, ::buffer<double, 1> *b_log_sqrt, double T );
double log_sqrt ( double T, double eigen );
double max_solve_3rd_degree_eq ( double a, double b, double c, double d);
double max_eigen_2D ( double A10, double A11, double A20, double A21 );
double max_eigen_3D ( double A10, double A11, double A12, double A20, double A21, double A22, double A30, double A31, double A32 );

