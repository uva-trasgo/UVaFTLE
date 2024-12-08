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
#include "preprocess.h"

void read_coordinates ( char *filename, int nDim, int nPoints, double *coords )
{
	int ip, d, check_EOF;
	char buffer[255];
	FILE *file;

	// Open file
	file = fopen( filename, "r" );

	// First element must be nPoints
	check_EOF = fscanf(file, "%s", buffer);
	if ( check_EOF == EOF )
	{
		fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" );
		exit(-1);
	}

	// Rest of read elements will be points' coordinates
	for ( ip = 0; ip < nPoints; ip++ )
	{
		for ( d = 0; d < nDim; d++ )
		{
			check_EOF = fscanf(file, "%s", buffer);
			if ( check_EOF == EOF )
			{
				fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" );
				exit(-1);
			}
			coords[ip * nDim + d] = atof(buffer);
		}
	}

	// Close file
	fclose(file);
}

void read_faces ( char *filename, int nDim, int nVertsPerFace, int nFaces, int *faces )
{
   int iface, ielem, check_EOF;
   char buffer[255];
   FILE *file;

   // Open file
   file = fopen( filename, "r" );

   // First element must be nFaces
   check_EOF = fscanf(file, "%s", buffer);
   if ( check_EOF == EOF )
   {
      fprintf( stderr, "Error: Unexpected EOF in read_faces\n" );
      exit(-1);
   }

   // Rest of read elements will be faces points' indices
   for ( iface = 0; iface < nFaces; iface++ )
   {
      for ( ielem = 0; ielem < nVertsPerFace; ielem++ )
      {
	 check_EOF = fscanf(file, "%s", buffer);
	 if ( check_EOF == EOF )
	 {
	    fprintf( stderr, "Error: Unexpected EOF in read_faces\n" );
	    exit(-1);
	 }
	 faces[iface * nVertsPerFace + ielem] = atoi(buffer);
      }
   }

   // Close file
   fclose(file);
}

void read_flowmap ( char *filename, int nDims, int nPoints, double *flowmap )
{
   int ip, idim, check_EOF;
   char buffer[255];
   FILE *file;

   // Open file
   file = fopen( filename, "r" );

   // Set velocity vectors space
   for ( ip = 0; ip < nPoints; ip++ )
   {
      for ( idim = 0; idim < nDims; idim++ )
	  {
		check_EOF = fscanf(file, "%s", buffer);
		if ( check_EOF == EOF )
		{
			fprintf( stderr, "Error: Unexpected EOF in read_flowmap\n" );
			exit(-1);
		}
		flowmap[ip * nDims + idim] = atof(buffer);
	  }
   }

   // Close file
   fclose(file);
}

void create_nFacesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint )
{
	int ip, iface, ipf;
	for ( ip = 0; ip < nPoints; ip++ )
	{
		nFacesPerPoint[ip] = 0;
	}
	for ( iface = 0; iface < nFaces; iface++ )
	{
		for ( ipf = 0; ipf < nVertsPerFace; ipf++ )
		{
			ip = faces[iface * nVertsPerFace + ipf];
			nFacesPerPoint[ip] = nFacesPerPoint[ip] + 1;
		}
	}
	for ( ip = 1; ip < nPoints; ip++ )
	{
		nFacesPerPoint[ip] = nFacesPerPoint[ip] + nFacesPerPoint[ip-1];
	}	
}

__global__ void create_facesPerPoint_vector_opt( int nFaces, int nVertsPerFace, const int *faces, const int *startIndices, int *writeIndices, int *facesPerPoint )
{
    int iface = blockIdx.x * blockDim.x + threadIdx.x;
    if (iface < nFaces)
    {
        // Loop over vertices of the face
        for (int ipf = 0; ipf < nVertsPerFace; ipf++)
        {
            int point = faces[iface * nVertsPerFace + ipf];

            // Atomically increment the write index for this point
            int index = atomicAdd(&writeIndices[point], 1);

            // Compute the position to write in facesPerPoint
            facesPerPoint[startIndices[point] + index] = iface;
        }
    }
}

