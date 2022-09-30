#ifndef STRUCTS_H
#define STRUCTS_H
typedef struct Face {
   int      index;      /* index to mesh->faces structure */
   int     *vertices;   /* index to mesh->points structure */
} face_t;

typedef struct Point {
   int       index; // Point index in mesh->points
   double   *coordinates;
   int       nfaces;
   int      *faces;
} point_t;

typedef struct Mesh {
   int       nDim;
   int       nPoints;
   int       nFaces;
   int       nVertsPerFace;
   point_t  *points;
   face_t   *faces;
} mesh_t;
#endif