#ifndef THTENSOR_CUDA_JNI_H
#define THTENSOR_CUDA_JNI_H

#include "TH/TH.h"
#include "THC/THC.h"

#ifdef THTensor_
#undef THTensor_
#endif
#ifdef THStorage_
#undef THStorage_
#endif
#ifdef real
#undef real
#endif
#ifdef accreal
#undef accreal
#endif
#ifdef THTensor
#undef THTensor
#endif
#ifdef THStorage
#undef THStorage
#endif
typedef float real;
typedef double accreal;
typedef THCudaTensor THTensor;
typedef THCudaStorage THStorage;
#define THTensor_(x) THCudaTensor_##x
#define THStorage_(x) THCudaStorage_##x

#endif
