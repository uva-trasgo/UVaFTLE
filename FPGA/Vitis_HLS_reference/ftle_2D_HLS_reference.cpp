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

////////////////////////////////////////////////// 
////////////////////////////////////////////////// 
//                                              //
//  File:   ftle_2D_reference.cpp               //
//                                              //
//  Authors:                                    //
//          Roberto Rodríguez Osorio            //
//          University of A Coruña, Spain       //
//          ORCID:  0000-0001-8768-2240         //
//          Manuel de Castro Caballero          //
//          University of Valladolid, Spain     //
//          ORCID:  0000-0003-3080-5136         //
//                                              //
////////////////////////////////////////////////// 
////////////////////////////////////////////////// 

#include <iostream>
#include "ap_int.h"
#include "math.h"

using namespace std;

// double arithmetic2D performs the actual calculations for the 2D case
// The 4 required coordinates: X0, X1, Y2 and Y3 are provided by the caller function,
// as well as the 8 flow values 

double arithmetic2D(
		double X0,
		double X1,
		double Y2,
		double Y3,
		double flowX0,
		double flowX1,
		double flowX2,
		double flowX3,
		double flowY0,
		double flowY1,
		double flowY2,
		double flowY3,
		const double factorT
){
#pragma HLS INTERFACE ap_ctrl_none port=X0
#pragma HLS INTERFACE ap_none port=X0
#pragma HLS INTERFACE ap_ctrl_none port=flowX0
#pragma HLS INTERFACE ap_none port=flowX0
#pragma HLS INTERFACE ap_ctrl_none port=flowY0
#pragma HLS INTERFACE ap_none port=flowY0
#pragma HLS INTERFACE ap_ctrl_none port=X1
#pragma HLS INTERFACE ap_none port=X1
#pragma HLS INTERFACE ap_ctrl_none port=flowX1
#pragma HLS INTERFACE ap_none port=flowX1
#pragma HLS INTERFACE ap_ctrl_none port=flowY1
#pragma HLS INTERFACE ap_none port=flowY1
#pragma HLS INTERFACE ap_ctrl_none port=Y2
#pragma HLS INTERFACE ap_none port=Y2
#pragma HLS INTERFACE ap_ctrl_none port=flowX2
#pragma HLS INTERFACE ap_none port=flowX2
#pragma HLS INTERFACE ap_ctrl_none port=flowY2
#pragma HLS INTERFACE ap_none port=flowY2
#pragma HLS INTERFACE ap_ctrl_none port=Y3
#pragma HLS INTERFACE ap_none port=Y3
#pragma HLS INTERFACE ap_ctrl_none port=flowX3
#pragma HLS INTERFACE ap_none port=flowX3
#pragma HLS INTERFACE ap_ctrl_none port=flowY3
#pragma HLS INTERFACE ap_none port=flowY3

#pragma HLS pipeline II = 1

	double denom_x, denom_y, gra10, gra20, gra11, gra21;
	double mat10, mat20, mat11, mat21;
	double A10, A11, A20, A21;

	double sq10, sq21, p3, p4;
	double sum, root, max, logarithm;

	// coordinates and flows for non-existing points have already been put to zero
	// therefore, this function is completely regular regardless of how many neighors 
	// a given point has. 

	denom_x = X1 - X0;
	denom_y = Y3 - Y2;

	gra10 = (flowX1 - flowX0) / denom_x;
	gra20 = (flowX3 - flowX2) / denom_y;
	gra11 = (flowY1 - flowY0) / denom_x;
	gra21 = (flowY3 - flowY2) / denom_y;

	mat10 = gra10 * gra10 + gra11 * gra11;
	mat11 = gra10 * gra20 + gra11 * gra21;
	mat20 = gra20 * gra10 + gra21 * gra11;
	mat21 = gra20 * gra20 + gra21 * gra21;

	A10 = mat10 * mat10 + mat11 * mat11;
	A11 = mat10 * mat20 + mat11 * mat21;
	A20 = mat20 * mat10 + mat21 * mat11;
	A21 = mat20 * mat20 + mat21 * mat21;

	sq21 = A21 * A21;
	sq10 = A10 * A10;
	p3 = A10 * A21;
	p4 = A11 * A20;

	sum = sq21 + sq10 - 2 * p3 + 4 * p4;

	root = sqrt(sum);

	max = (A21 + A10 + root) / 2;

	logarithm = log(max);
	return logarithm * factorT;
	
	// originally: 	return log( sqrt(max) ) / T;
	// as T is constant, factorT is defined as T / 2
	// taking advantage of: log( sqrt(max) ) = 1/2 * log(max)
	// and multiplying by factorT instead of the more costly square root and quotient. 
}



// ftle_2d() gets the indexes of the 4 neighbors of the current point (vec0, vec1, vec2, vec3).
// as well as pointers to 2 coordinates and 2 flow arrays: coordsX, flowmapX, coordsY, and flowmapY.
// ftle_2d() just reads the actual coordinates and flow using indexes as addresses. 
// A non existing neighbor is signaled as (-1), and ftle_2d() will issue zeros as coord and flowmap values. 
// arithmetic2D() is invoqued with those values. 


double ftle_2d(int vec0, int vec1, int vec2, int vec3, double *coordsX, double *flowmapX, double *coordsY, double *flowmapY, double factorT){

#pragma HLS INTERFACE ap_ctrl_none port=vec0
#pragma HLS INTERFACE ap_none port=vec0
#pragma HLS INTERFACE ap_ctrl_none port=vec1
#pragma HLS INTERFACE ap_none port=vec1
#pragma HLS INTERFACE ap_ctrl_none port=vec2
#pragma HLS INTERFACE ap_none port=vec2
#pragma HLS INTERFACE ap_ctrl_none port=vec3
#pragma HLS INTERFACE ap_none port=vec3

#pragma HLS INTERFACE m_axi port = coordsX depth = 1000000000 bundle = gmemCoordsX
#pragma HLS INTERFACE m_axi port = flowmapX depth = 1000000000 bundle = gmemflomapX
#pragma HLS INTERFACE m_axi port = coordsY depth = 1000000000 bundle = gmemCoordsY
#pragma HLS INTERFACE m_axi port = flowmapY depth = 1000000000 bundle = gmemFlowmapY

#pragma HLS pipeline II = 1

	double X0, X1, Y2, Y3, flowX0, flowX1, flowX2, flowX3, flowY0, flowY1, flowY2, flowY3;

	X0 = vec0 != -1 ? coordsX[vec0] : 0.0;
	X1 = vec1 != -1 ? coordsX[vec1] : 0.0;
	Y2 = vec2 != -1 ? coordsY[vec2] : 0.0;
	Y3 = vec3 != -1 ? coordsY[vec3] : 0.0;

	flowX0 = vec0 != -1 ? flowmapX[vec0] : 0.0;
	flowX1 = vec1 != -1 ? flowmapX[vec1] : 0.0;
	flowX2 = vec2 != -1 ? flowmapX[vec2] : 0.0;
	flowX3 = vec3 != -1 ? flowmapX[vec3] : 0.0;

	flowY0 = vec0 != -1 ? flowmapY[vec0] : 0.0;
	flowY1 = vec1 != -1 ? flowmapY[vec1] : 0.0;
	flowY2 = vec2 != -1 ? flowmapY[vec2] : 0.0;
	flowY3 = vec3 != -1 ? flowmapY[vec3] : 0.0;

	return arithmetic2D(X0, X1, Y2, Y3, flowX0, flowX1, flowX2, flowX3, flowY0, flowY1, flowY2, flowY3, factorT);
}
