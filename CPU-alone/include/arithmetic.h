#include "ftle.h"

void compute_gradient_2D ( int ip, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *log_sqrt, double T );
void compute_gradient_3D ( int ip, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *log_sqrt, double T );
double log_sqrt ( double T, double eigen );
double max_solve_3rd_degree_eq ( double a, double b, double c, double d);
double max_eigen_2D ( double A10, double A11, double A20, double A21 );
double max_eigen_3D ( double A10, double A11, double A12, double A20, double A21, double A22, double A30, double A31, double A32 );

