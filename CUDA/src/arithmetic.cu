/*
 *            UVaFTLE 1.0: Lagrangian finite time 
 *		    Lyapunov exponent extraction 
 *		    for fluid dynamic applications
 *
 *    Copyright (C) 2023, 2024 Rocío Carratalá-Sáez et. al.
 *    This file is part of the UVaFTLE application.
 *
 *  UVaFTLE is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  UVaFTLE is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with UVaFTLE.  If not, see <http://www.gnu.org/licenses/>.
 */ 
 
#include <stdio.h>
#include "arithmetic.h"

__global__ void gpu_compute_gradient_2D(int nPoints, int offset, int faces_offset, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *d_logSqrt, double T) 
{
	int gpu_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (gpu_id < nPoints){
		int th_id = gpu_id + offset;
		int nDim = 2; 
		int iface, nFaces, idxface, ivert;
		int closest_points_0 = -1;
		int closest_points_1 = -1;
		int closest_points_2 = -1;
		int closest_points_3 = -1;
		int count = 0;
	
		int ivertex;
		double denom_x, denom_y;
		double gra10, gra11, gra20, gra21;
		double ftle_matrix[4], d_W_ei[2];
		nFaces  = (th_id == 0) ? nFacesPerPoint[th_id] : nFacesPerPoint[th_id] - nFacesPerPoint[th_id-1];
		/* Find 4 closest points */
		int base_face = (th_id == 0) ? 0 : nFacesPerPoint[th_id-1] - faces_offset;
		for ( iface = 0; (iface < nFaces) && (count < 4); iface++ )
		{
			idxface =  facesPerPoint[base_face + iface];
			for ( ivert = 0; (ivert < nVertsPerFace) && (count < 4); ivert++ )
			{
				ivertex = faces[idxface * nVertsPerFace + ivert];
				if ( ivertex != th_id )
				{
					/* (i-1, j) */
					if ( (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1]) && (coords[ivertex * nDim] < coords[th_id * nDim]) )
					{
						if (closest_points_0 == -1)
						{
							closest_points_0 = ivertex;
							count++;
						}
					}
					else
					{
						/* (i+1, j) */
						if ( (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1]) && (coords[ivertex * nDim] > coords[th_id * nDim]) )
						{
							if (closest_points_1 == -1)
							{
								closest_points_1 = ivertex;
								count++;
							}
						}
						else
						{
							/* (i, j-1) */
							if ( (coords[ivertex * nDim] == coords[th_id * nDim]) && (coords[ivertex * nDim + 1] < coords[th_id * nDim + 1]) ) 
							{
								if (closest_points_2 == -1)
								{
									closest_points_2 = ivertex;
									count++;
								}
							}
							else
							{
								/* (i, j+1) */
								if ( (coords[ivertex * nDim] == coords[th_id * nDim]) && (coords[ivertex * nDim + 1] > coords[th_id * nDim + 1]) )
								{
									if (closest_points_3 == -1)
									{
										closest_points_3 = ivertex;
										count++;
									}
								}
							}
						}
					}			
				}
			}
		}
		if ( count == 4 )
		{
			/* NOTE: take care with denom_x and denom_y zero */
			denom_x = coords[ closest_points_1 * nDim ]	- coords[ closest_points_0 * nDim ]; 
			denom_y = coords[ closest_points_3 * nDim + 1]	- coords[ closest_points_2 * nDim + 1];
			gra10 = ( flowmap[ closest_points_1 * nDim ]	- flowmap[ closest_points_0 * nDim ] ) / denom_x; 			
			gra20 = ( flowmap[ closest_points_3 * nDim ]	- flowmap[ closest_points_2 * nDim ] ) / denom_y;
			gra11 = ( flowmap[ closest_points_1 * nDim + 1 ] - flowmap[ closest_points_0 * nDim + 1 ] ) / denom_x;
			gra21 = ( flowmap[ closest_points_3 * nDim + 1 ] - flowmap[ closest_points_2 * nDim + 1 ] ) / denom_y;
		}
		else
		{
			if ( count == 3 )
			{
				// (i-1, j) and (i+1, j) 
				if ( ( closest_points_0 > -1 ) && ( closest_points_1 > -1 ) )
				{
					denom_x = coords[ closest_points_1 * nDim ]	- coords[ closest_points_0 * nDim ]; 
					gra10 = ( flowmap[ closest_points_1 * nDim ] - flowmap[ closest_points_0 * nDim ] ) / denom_x;					
					gra20 = 1; 
					gra11 = ( flowmap[ closest_points_1 * nDim + 1 ] - flowmap[ closest_points_0 * nDim + 1 ] ) / denom_x;
					gra21 = 1;
				}
				// (i-1, j) and (i+1, j) 
				else
				{
					denom_y = coords[ closest_points_3 * nDim + 1] - coords[ closest_points_2 * nDim + 1];
					gra10 = 1; 
					gra20 = ( flowmap[ closest_points_3 * nDim ]	 - flowmap[ closest_points_2 * nDim ] )	 / denom_y;
					gra11 = 1;
					gra21 = ( flowmap[ closest_points_3 * nDim + 1 ] - flowmap[ closest_points_2 * nDim + 1 ] ) / denom_y;
				}
			}
			else
			{
				gra10 = 1;
				gra20 = 1;
				gra11 = 1;
				gra21 = 1;
			}
		}

		ftle_matrix[0] = gra10 * gra10 + gra11 * gra11;
		ftle_matrix[1] = gra10 * gra20 + gra11 * gra21;
		ftle_matrix[2] = gra20 * gra10 + gra21 * gra11;
		ftle_matrix[3] = gra20 * gra20 + gra21 * gra21;
		gra10 = ftle_matrix[0];
		gra11 = ftle_matrix[1];
		gra20 = ftle_matrix[2];
		gra21 = ftle_matrix[3];

		ftle_matrix[0] = gra10 * gra10 + gra11 * gra11;
		ftle_matrix[1] = gra10 * gra20 + gra11 * gra21;
		ftle_matrix[2] = gra20 * gra10 + gra21 * gra11;
		ftle_matrix[3] = gra20 * gra20 + gra21 * gra21;
		double A10 = ftle_matrix[0];
		double A11 = ftle_matrix[1];
		double A20 = ftle_matrix[2];
		double A21 = ftle_matrix[3];
		double sq = sqrt(A21 * A21 + A10 * A10 - 2 * (A10 * A21) + 4 * (A11 * A20));
		d_W_ei[0] = (A21 + A10 + sq) / 2;
		d_W_ei[1] = (A21 + A10 - sq) / 2; 

		//---------------- max---sqrt---log
		double max = d_W_ei[0];	 
		if (d_W_ei[1] > max ) max = d_W_ei[1];

		max = sqrt(max);
		max = log (max);
		d_logSqrt[gpu_id] = max / T;
	}
}

__global__ void gpu_compute_gradient_3D (int nPoints, int offset, int faces_offset, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *d_logSqrt, double T) //double *gra1, double *gra2 )
{
	int gpu_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (gpu_id < nPoints){
		int th_id = gpu_id + offset;
		int nDim = 3; 
		int iface, nFaces, idxface, ivert;
		int closest_points_0 = -1;
		int closest_points_1 = -1;
		int closest_points_2 = -1;
		int closest_points_3 = -1;
		int closest_points_4 = -1;
		int closest_points_5 = -1;
		int count = 0;
	
		int ivertex;
		double denom_x, denom_y, denom_z;
		double ftle_matrix[9];
		double gra10, gra11, gra12, gra20, gra21, gra22, gra30, gra31, gra32;
		nFaces  = (th_id == 0) ? nFacesPerPoint[th_id] : nFacesPerPoint[th_id] - nFacesPerPoint[th_id-1];
		/* Find 6 closest points */
		int base_face = (th_id == 0) ? 0 : nFacesPerPoint[th_id-1] - faces_offset;
		for ( iface = 0; (iface < nFaces) && (count < 6); iface++ )	{
			idxface =  facesPerPoint[base_face + iface];
			for ( ivert = 0; (ivert < nVertsPerFace) && (count < 6); ivert++ )
			{
				ivertex = faces[idxface * nVertsPerFace + ivert];
				if ( ivertex != th_id ){
					/* (i-1, j, k) */
					if (   (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1])
						&& (coords[ivertex * nDim + 2] == coords[th_id * nDim + 2]) 
						&& (coords[ivertex * nDim]	 <  coords[th_id * nDim]) )
					{
						if (closest_points_0 == -1)
						{
							closest_points_0 = ivertex;
							count++;
						}
					}
					else
					{
						/* (i+1, j, k) */
						if (   (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1]) 
							&& (coords[ivertex * nDim + 2] == coords[th_id * nDim + 2]) 
							&& (coords[ivertex * nDim]	 >  coords[th_id * nDim]) )
						{
							if (closest_points_1 == -1)
							{
								closest_points_1 = ivertex;
								count++;
							}
						}
						else
						{
							/* (i, j-1, k) */
							if (   (coords[ivertex * nDim]	 == coords[th_id * nDim]) 
								&& (coords[ivertex * nDim + 2] == coords[th_id * nDim + 2]) 
								&& (coords[ivertex * nDim + 1] <  coords[th_id * nDim + 1]) ) 
							{
								if (closest_points_2 == -1)
								{
									closest_points_2 = ivertex;
									count++;
								}
							}
							else
							{
								/* (i, j+1, k) */
								if (   (coords[ivertex * nDim]	 == coords[th_id * nDim]) 
									&& (coords[ivertex * nDim + 2] == coords[th_id * nDim + 2]) 
									&& (coords[ivertex * nDim + 1] >  coords[th_id * nDim + 1]) )
								{
									if (closest_points_3 == -1)
									{
										closest_points_3 = ivertex;
										count++;
									}
								}
								else
								{
									/* (i, j, k-1) */
									if (   (coords[ivertex * nDim]	 == coords[th_id * nDim]) 
										&& (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1]) 
										&& (coords[ivertex * nDim + 2] <  coords[th_id * nDim + 2]) )
									{
										if (closest_points_4 == -1)
										{
											closest_points_4 = ivertex;
											count++;
										}
									}
									else
									{
										/* (i, j, k+1) */
										if (   (coords[ivertex * nDim]	 == coords[th_id * nDim]) 
											&& (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1]) 
											&& (coords[ivertex * nDim + 2] >  coords[th_id * nDim + 2]) )
										{
											if (closest_points_5 == -1)
											{
												closest_points_5 = ivertex;
												count++;
											}
										}
									}
								}
							}
						}
					}			
				}
			}
		}
		if ( count == 6 ){
			/* NOTE: take care with denom_x and denom_y zero */
			denom_x = coords[ closest_points_1 * nDim ]	- coords[ closest_points_0 * nDim ]; 
			denom_y = coords[ closest_points_3 * nDim + 1] - coords[ closest_points_2 * nDim + 1];
			denom_z = coords[ closest_points_5 * nDim + 2] - coords[ closest_points_4 * nDim + 2];

			gra10 = ( flowmap[ closest_points_1 * nDim ] - flowmap[ closest_points_0 * nDim ] ) / denom_x; 			
			gra11 = ( flowmap[ closest_points_3 * nDim ] - flowmap[ closest_points_2 * nDim ] ) / denom_y;
		   	gra12 = ( flowmap[ closest_points_5 * nDim ] - flowmap[ closest_points_4 * nDim ] ) / denom_z;

			gra20 = ( flowmap[ closest_points_1 * nDim + 1] - flowmap[ closest_points_0 * nDim + 1] ) / denom_x; 			
			gra21 = ( flowmap[ closest_points_3 * nDim + 1] - flowmap[ closest_points_2 * nDim + 1] ) / denom_y;
		   	gra22 = ( flowmap[ closest_points_5 * nDim + 1] - flowmap[ closest_points_4 * nDim + 1] ) / denom_z;

			gra30 = ( flowmap[ closest_points_1 * nDim + 2] - flowmap[ closest_points_0 * nDim + 2] ) / denom_x; 			
			gra31 = ( flowmap[ closest_points_3 * nDim + 2] - flowmap[ closest_points_2 * nDim + 2] ) / denom_y;
		   	gra32 = ( flowmap[ closest_points_5 * nDim + 2] - flowmap[ closest_points_4 * nDim + 2] ) / denom_z;
		}
		else
		{
			gra10 = 1;
			gra11 = 1;
			gra12 = 1;
			gra20 = 1;
			gra21 = 1;
			gra22 = 1;
			gra30 = 1;
			gra31 = 1;
			gra32 = 1;
		}

		/* Tens */
		ftle_matrix[0] = gra10 * gra10 + gra20 * gra20 + gra30 * gra30;
		ftle_matrix[1] = gra10 * gra11 + gra20 * gra21 + gra30 * gra31;
		ftle_matrix[2] = gra10 * gra12 + gra20 * gra22 + gra30 * gra32;
		ftle_matrix[3] = ftle_matrix[1];
		ftle_matrix[4] = gra11 * gra11 + gra21 * gra21 + gra31 * gra31;
		ftle_matrix[5] = gra11 * gra12 + gra11 * gra22 + gra31 * gra32;
		ftle_matrix[6] = ftle_matrix[2];
		ftle_matrix[7] = ftle_matrix[5];
		ftle_matrix[8] = gra12 * gra12 + gra22 * gra22 + gra32 * gra32;

		// Store copy to later multiply by transpose 
		gra10 = ftle_matrix[0];
		gra11 = ftle_matrix[1];
		gra12 = ftle_matrix[2];
		gra20 = ftle_matrix[3];
		gra21 = ftle_matrix[4];
		gra22 = ftle_matrix[5];
		gra30 = ftle_matrix[6];
		gra31 = ftle_matrix[7];
		gra32 = ftle_matrix[8];

		// Matrix mult 
		double A10 = gra10 * gra10 + gra11 * gra11 + gra12 * gra12;
		double A11 = gra10 * gra20 + gra11 * gra21 + gra12 * gra22;
		double A12 = gra10 * gra30 + gra11 * gra31 + gra12 * gra32;
		double A20 = gra20 * gra10 + gra21 * gra11 + gra22 * gra12;
		double A21 = gra20 * gra20 + gra21 * gra21 + gra22 * gra22;
		double A22 = gra20 * gra30 + gra21 * gra31 + gra22 * gra32;
		double A30 = gra30 * gra10 + gra31 * gra11 + gra32 * gra12;
		double A31 = gra30 * gra20 + gra31 * gra21 + gra32 * gra22;
		double A32 = gra30 * gra30 + gra31 * gra31 + gra32 * gra32;


		double a = -1;
		double b = A10 + A21 + A32;
		double c = A12 * A30 + A22 * A31 + A11 * A20 - A10 * A21 - A10 * A32 - A21 * A32;
		double d = A10 * A21 * A32 + A11 * A22 * A30 + A12 * A20 * A31 - A10 * A22 * A31 - A11 * A20 * A32 - A12 * A21 * A30;
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
			double sqA = sqrt(A);
			double sq3 = sqrt(3.0);
			double T   = (2*A*b-3*a*B) / (2*A*sqA);
			double _xt = acos(T);
			double xt  = _xt/3;
			x1	 = (-b-2*sqA*cos(xt)) / (3*a);
			x2	 = (-b+sqA*(cos(xt)+sq3*sin(xt)))/(3*a);
			x3	 = (-b+sqA*(cos(xt)-sq3*sin(xt)))/(3*a);
		}
		double max = x1;
		if (x2 > max ) max = x2;
		if (x3 > max ) max = x3;
				
		max = sqrt(max);
		max = log (max);
		d_logSqrt[gpu_id] = max / T;
	}
}
