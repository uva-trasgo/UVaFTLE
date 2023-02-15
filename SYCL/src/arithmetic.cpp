#include "arithmetic.h"
#include "math.h"

double max_solve_3rd_degree_eq ( double a, double b, double c, double d)
{
	double x1, x2, x3;
	double A   = b*b - 3*a*c;
	double B   = b*c - 9*a*d;
	double C   = c*c - 3*b*d;
	double del = B*B - 4*A*C;
	if ( A == B && A == 0 )
	{
		x1 = x2 = x3 = -b/(3*a);
	}
	else if(del==0)
	{
		x1 = -b/a+B/A;
		x2 = x3 = (-B/A)/2;
	}
	else if(del<0)
	{
		double T   = (2*A*b-3*a*B) / (2*A*sqrt(A));
		double _xt = acos(T);
		double xt  = _xt/3;
		x1         = (-b-2*sqrt(A)*cos(xt)) / (3*a);
		x2         = (-b+sqrt(A)*(cos(xt)+sqrt(3)*sin(xt)))/(3*a);
		x3         = (-b+sqrt(A)*(cos(xt)-sqrt(3)*sin(xt)))/(3*a);
	}
	double max = ( x1 > x2 ) ? x1 : x2;
	return ( max > x3 ) ? max : x3;
}

void compute_gradient_2D ( int ip, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *log_sqrt, double T )
{

}

void compute_gradient_3D ( int ip, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *log_sqrt, double T)
{
	compute_gradient_2D (ip, nVertsPerFace, coords, flowmap, faces, nFacesPerPoint, facesPerPoint, log_sqrt, T);

}