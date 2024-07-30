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

#ifndef OPENCL_HELPER_H
#define OPENCL_HELPER_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#ifdef OPENCL_DEBUG
#define OPENCL_CHECK(operation)                                     \
	err = operation;                                                \
	if (err != CL_SUCCESS)                                          \
		fprintf(stderr, "[OpenCL error]: Error number %d @%s:%d\n", \
		        err, __FILE__, __LINE__);
#define OPENCL_CHECK_ERROR(err)                                     \
	if (err != CL_SUCCESS)                                          \
		fprintf(stderr, "[OpenCL error]: Error number %d @%s:%d\n", \
		        err, __FILE__, __LINE__);
#define MALLOC_CHECK(ptr)                                           \
	if (ptr == NULL)                                                \
		fprintf(stderr, "[malloc error]: Couldn't allocate buffer " \
		                "\"%s\" @%s:%d\n",                          \
		        #ptr, __FILE__, __LINE__);
#else // !OPENCL_DEBUG
#define OPENCL_CHECK(operation) \
	operation
#define OPENCL_CHECK_ERROR(err)
#define MALLOC_CHECK(ptr)
#endif // OPENCL_DEBUG

cl_context cl_context_and_device_for_platform(const char *platform, cl_device_id *device_id);
cl_kernel *cl_kernels_from_names(const char *file_path, cl_context context, cl_device_id device_id, const int n_kernels, const char **entrypoints);

#endif // OPENCL_KERNEL_H
