#include "preprocess.h"
using namespace cl::sycl;

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

void create_facesPerPoint_vector (queue* q, int nDim, int nPoints, int offset, int nFaces, int nVertsPerFace, cl::sycl::buffer<int, 1> *b_faces, cl::sycl::buffer<int, 1> *b_nFacesPerPoint, cl::sycl::buffer<int, 1> *b_facesPerPoint)
{
	q->submit([&](handler &h){
		auto faces = b_faces->get_access<access::mode::read>(h);
		auto nFacesPerPoint = b_nFacesPerPoint->get_access<access::mode::read>(h);
		auto facesPerPoint = b_facesPerPoint->get_access<access::mode::discard_write>(h);
		h.parallel_for<class preprocess> (range<1>{static_cast<size_t>(nPoints)}, [=](id<1> i){
			int ip, count, iface, ipf, nFacesP, iFacesP, faces_offset;   
			count = 0;
			ip= i[0]+offset;	
			faces_offset = (offset == 0) ? 0 :  nFacesPerPoint[offset-1];
			iFacesP = ( ip == 0 ) ? 0 : nFacesPerPoint[ip-1];
			nFacesP = ( ip == 0 ) ? nFacesPerPoint[ip] : nFacesPerPoint[ip] - nFacesPerPoint[ip-1];
			for ( iface = 0; ( iface < nFaces ) && ( count < nFacesP ); iface++ ){     
				for ( ipf = 0; ipf < nVertsPerFace; ipf++ ){       
					if ( faces[iface * nVertsPerFace + ipf] == ip ){
						facesPerPoint[iFacesP - faces_offset + count] = iface;
						count++;
					}
				}
			}
		}); /*End parallel for*/
	}); /*End submit*/	
}


void create_facesPerPoint_vector2 (queue* q, int nDim, int nPoints, int nFaces, int nVertsPerFace, cl::sycl::buffer<int, 1> *b_faces, cl::sycl::buffer<int, 1> *b_nFacesPerPoint, cl::sycl::buffer<int, 1> *b_facesPerPoint)
{
	q->submit([&](handler &h){
		auto faces = b_faces->get_access<access::mode::read>(h);
		auto nFacesPerPoint = b_nFacesPerPoint->get_access<access::mode::read>(h);
		auto facesPerPoint = b_facesPerPoint->get_access<access::mode::discard_write>(h);
		h.parallel_for<class preprocess> (range<1>{static_cast<size_t>(nPoints)}, [=](id<1> i){
			int ip, count, iface, ipf, nFacesP, iFacesP;   
			count = 0;
			ip= i[0];	
			iFacesP = ( ip == 0 ) ? 0 : nFacesPerPoint[ip-1];
			nFacesP = ( ip == 0 ) ? nFacesPerPoint[ip] : nFacesPerPoint[ip] - nFacesPerPoint[ip-1];
			for ( iface = 0; ( iface < nFaces ) && ( count < nFacesP ); iface++ ){     
				for ( ipf = 0; ipf < nVertsPerFace; ipf++ ){       
					if ( faces[iface * nVertsPerFace + ipf] == ip ){
						facesPerPoint[iFacesP + count] = iface;
						count++;
					}
				}
			}
		}); /*End parallel for*/
	}); /*End submit*/	
}
