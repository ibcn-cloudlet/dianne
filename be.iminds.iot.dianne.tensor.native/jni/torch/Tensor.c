/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
#include "be_iminds_iot_dianne_tensor_Tensor.h"
#include "TensorLoader.h"

THGenerator* generator = 0;


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_init
  (JNIEnv * env, jobject t, jfloatArray data, jintArray dims){
	THTensor * tensor;

	if(dims == NULL){
		tensor = THTensor_(new)(
#ifdef CUDA
				state
#endif
		);
		return (long) tensor;
	}

	jsize noDims = (*env)->GetArrayLength(env, dims);

	jint *d = (*env)->GetIntArrayElements(env, dims, 0);
	if(noDims==1){
		tensor = THTensor_(newWithSize1d)(
#ifdef CUDA
				state,
#endif
				d[0]);
	} else if(noDims==2){
		tensor = THTensor_(newWithSize2d)(
#ifdef CUDA
				state,
#endif
				d[0], d[1]);
	} else if(noDims==3){
		tensor = THTensor_(newWithSize3d)(
#ifdef CUDA
				state,
#endif
				d[0], d[1], d[2]);
	} else if(noDims==4){
		tensor = THTensor_(newWithSize4d)(
#ifdef CUDA
				state,
#endif
				d[0], d[1], d[2], d[3]);
	} // for now only support up to 4D tensors...
	(*env)->ReleaseIntArrayElements(env, dims, d, 0);

	if(data != NULL){
		jsize len = (*env)->GetArrayLength(env, data);
		jfloat * floats = (*env)->GetFloatArrayElements(env, data, 0);

#ifdef CUDA
		cudaMemcpy(THTensor_(data)(state, tensor), floats, len*sizeof(real), cudaMemcpyHostToDevice);
	    THCudaCheck(cudaGetLastError());
#else
		jfloat* src_ptr = floats;
		real* dst_ptr = THTensor_(data)(tensor);
		int i =0;
		for(i=0;i<len;i++){
			*(dst_ptr++) = *(src_ptr++);
		}
#endif

		(*env)->ReleaseFloatArrayElements(env, data, floats, 0);
	}

	return (long) tensor;
}



JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_free
  (JNIEnv * env, jobject t){
	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			getTensor(env, t));
}


JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_dim
  (JNIEnv * env, jobject t){
	THTensor* tensor = getTensor(env, t);
    return tensor->nDimension;
}

JNIEXPORT jintArray JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_dims
  (JNIEnv * env, jobject t){
	THTensor* tensor = getTensor(env, t);

	long* ptr = tensor->size;

	int nDim = tensor->nDimension;
	jintArray result;
	result = (*env)->NewIntArray(env, nDim);
	if (result == NULL) {
	    return NULL;
	}
	int i;
	jint dims[nDim];
	for (i = 0; i < nDim; i++) {
	    dims[i] = *(ptr++);
	}
	(*env)->SetIntArrayRegion(env, result, 0, nDim, dims);
	return result;
}

JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_size__
  (JNIEnv * env, jobject t){
	THTensor* tensor = getTensor(env, t);

	return THTensor_(nElement)(
#ifdef CUDA
			state,
#endif
			tensor);
}

JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_size__I
  (JNIEnv * env, jobject t, jint d){
	THTensor* tensor = getTensor(env, t);
	return tensor->size[d];
}


JNIEXPORT jboolean JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_sameDim
 (JNIEnv * env, jobject t, jobject o){
	THTensor* tensor = getTensor(env, t);
	THTensor* tensor2 = getTensor(env, o);

	return THTensor_(isSameSizeAs)(
#ifdef CUDA
			state,
#endif
			tensor,
			tensor2);
}

JNIEXPORT jboolean JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_hasDim
 (JNIEnv * env, jobject t, jintArray dims){
	THTensor* tensor = getTensor(env, t);

	jsize nDim = (*env)->GetArrayLength(env, dims);

	if(tensor->nDimension != nDim){
		return 0;
	}

	jint *d = (*env)->GetIntArrayElements(env, dims, 0);
	long* ptr = tensor->size;
	int i;
	for(i=0; i< nDim ;i++){
		if(d[i] != *(ptr++)){
			return 0;
		}
	}

	(*env)->ReleaseIntArrayElements(env, dims, d, 0);
	return 1;
}


JNIEXPORT jboolean JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_equalsData
  (JNIEnv * env, jobject t, jobject o, jfloat threshold){
	THTensor* tensor = getTensor(env, t);
	THTensor* tensor2 = getTensor(env, o);

	THTensor* diff = THTensor_(new)(
#ifdef CUDA
			state
#endif
	);
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			diff, tensor);
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			diff, tensor2, -1.0f, tensor);
	THTensor_(abs)(
#ifdef CUDA
			state,
#endif
			diff, diff);

	real max = THTensor_(maxall)(
#ifdef CUDA
			state,
#endif
			diff);

	if(max <= threshold){
		return 1;
	} else {
		return 0;
	}
}



JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_reshape
  (JNIEnv * env, jobject t, jintArray dims){
	THTensor* tensor = getTensor(env, t);

	jsize noDims = (*env)->GetArrayLength(env, dims);

	jint *index = (*env)->GetIntArrayElements(env, dims, 0);
	if(noDims==1){
		THTensor_(resize1d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0]);
	} else if(noDims==2){
		THTensor_(resize2d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1]);
	} else if(noDims==3){
		THTensor_(resize3d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1], index[2]);
	} else if(noDims==4){
		THTensor_(resize4d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1], index[2], index[3]);
	} // for now only support up to 4D tensors...
	(*env)->ReleaseIntArrayElements(env, dims, index, 0);
}




JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_get___3I
  (JNIEnv * env, jobject t, jintArray d){
	float val;
	THTensor* tensor = getTensor(env, t);

	jsize noDims = (*env)->GetArrayLength(env, d);

	jint *index = (*env)->GetIntArrayElements(env, d, 0);
	if(noDims==1){
		val = THTensor_(get1d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0]);
	} else if(noDims==2){
		val = THTensor_(get2d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1]);
	} else if(noDims==3){
		val = THTensor_(get3d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1], index[2]);
	} else if(noDims==4){
		val = THTensor_(get4d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1], index[2], index[3]);
	} // for now only support up to 4D tensors...
	(*env)->ReleaseIntArrayElements(env, d, index, 0);

	return val;
}




JNIEXPORT jfloatArray JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_get__
  (JNIEnv * env, jobject t){

	THTensor* tensor = getTensor(env, t);

	real* ptr = THTensor_(data)(
#ifdef CUDA
			state,
#endif
			tensor);
	real* data = ptr;

#ifdef CUDA
	long bufferSize = (tensor->storage->size-tensor->storageOffset)*sizeof(real);
	real* buffer = malloc(bufferSize);
	cudaMemcpy(buffer, ptr, bufferSize, cudaMemcpyDeviceToHost);
    THCudaCheck(cudaGetLastError());
	data = buffer;
# endif

	// calculate actual size
	// (can be different of underlying data array in case of narrowed tensor)
	int i;
	int size = 1;
	for(i=0;i<tensor->nDimension;i++){
		size *= tensor->size[i];
	}

	real* narrowed;

	if(size==tensor->storage->size){
		narrowed = data;
	} else {
		// copy right data
		narrowed = malloc(size*sizeof(real));
		real* p = narrowed;

		// generic iterate over n-dim tensor data
		int index[tensor->nDimension];
		for(i=0;i<tensor->nDimension;i++){
			index[i] = 0;
		}
		int next = 0;

		while(next!=-1){
			*p++ = data[next];

			int incremented = 0;
			int dim = tensor->nDimension-1;
			while(incremented == 0){
				index[dim]++;
				if(index[dim]==tensor->size[dim]){
					index[dim] = 0;
					next-=tensor->stride[dim]*(tensor->size[dim]-1);
					dim--;
				} else {
					incremented = 1;
					next+= tensor->stride[dim];
				}
				if(dim<0){
					next = -1;
					incremented = 1;
				}
			}
		}
	}

	jfloatArray result;
	result = (*env)->NewFloatArray(env, size);
	if (result == NULL) {
	    return NULL;
	}
	(*env)->SetFloatArrayRegion(env, result, 0, size, narrowed);


#ifdef CUDA
	free(buffer);
#endif

	// free in case of narrowed data
	if(size!=tensor->storage->size){
		free(narrowed);
	}

	return result;
}




JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_set__F_3I
  (JNIEnv * env, jobject t, jfloat val, jintArray d){
	THTensor* tensor = getTensor(env, t);

	jsize noDims = (*env)->GetArrayLength(env, d);

	jint *index = (*env)->GetIntArrayElements(env, d, 0);
	if(noDims==1){
		THTensor_(set1d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], val);
	} else if(noDims==2){
		THTensor_(set2d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1], val);
	} else if(noDims==3){
		THTensor_(set3d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1], index[2], val);
	} else if(noDims==4){
		THTensor_(set4d)(
#ifdef CUDA
				state,
#endif
				tensor, index[0], index[1], index[2], index[3], val);
	} // for now only support up to 4D tensors...
	(*env)->ReleaseIntArrayElements(env, d, index, 0);
}



JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_set___3F
  (JNIEnv * env, jobject t, jfloatArray data){
	THTensor* tensor = getTensor(env, t);

	jsize len = (*env)->GetArrayLength(env, data);
	jfloat * floats = (*env)->GetFloatArrayElements(env, data, 0);

#ifdef CUDA
	cudaMemcpy(THTensor_(data)(state, tensor), floats, len*sizeof(real), cudaMemcpyHostToDevice);
    THCudaCheck(cudaGetLastError());
#else
	jfloat* src_ptr = floats;
	real* dst_ptr = THTensor_(data)(tensor);
	int i =0;
	for(i=0;i<len;i++){
		*(dst_ptr++) = *(src_ptr++);
	}
#endif

	(*env)->ReleaseFloatArrayElements(env, data, floats, 0);
}



JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_fill
  (JNIEnv * env, jobject t, jfloat val){
	THTensor* tensor = getTensor(env, t);

	THTensor_(fill)(
#ifdef CUDA
			state,
#endif
			tensor, val);
}



JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_rand
  (JNIEnv * env, jobject t){
	THTensor* tensor = getTensor(env, t);

#ifdef CUDA
	THTensor_(uniform)(state, tensor, 0, 1);
#else
	if(generator==0){
		generator = THGenerator_new();
	}

	THTensor_(uniform)(tensor, generator, 0, 1);
#endif
}



JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_randn
  (JNIEnv * env, jobject t){
	THTensor* tensor = getTensor(env, t);

#ifdef CUDA
	THTensor_(normal)(state, tensor, 0, 1);
#else
	if(generator==0){
		generator = THGenerator_new();
	}

	THTensor_(normal)(tensor, random, 0, 1);
#endif
}



JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_bernoulli
  (JNIEnv * env, jobject t, jfloat p){
	THTensor* tensor = getTensor(env, t);

#ifdef CUDA
	THTensor_(bernoulli)(state, tensor, p);
#else
	if(generator==0){
		generator = THGenerator_new();
	}

	THTensor_(bernoulli)(tensor, generator, p);
#endif
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_copyInto
  (JNIEnv * env, jobject t, jobject target){
	THTensor* tensor = getTensor(env, t);
	THTensor* tensor2 = getTensor(env, target);

	int size = THTensor_(nElement)(
#ifdef CUDA
		state,
#endif
		tensor);

	// resize if total size not equals
	int size2 = THTensor_(nElement)(
#ifdef CUDA
		state,
#endif
		tensor2);

	if(size != size2){
		THTensor_(resizeAs)(
#ifdef CUDA
				state,
#endif
				tensor2, tensor);
	}

#ifdef CUDA
	THTensor_(copyCuda)(state, tensor2, tensor);
#else
	THTensor_(copyFloat)(tensor2, tensor);
#endif

	return target == NULL ? createTensorObject(env, tensor2) : target;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_narrow
  (JNIEnv * env, jobject t, jint dim, jint index, jint size){
	THTensor* tensor = getTensor(env, t);

	THTensor* narrow = THTensor_(newNarrow)(
#ifdef CUDA
		state,
#endif
		tensor, dim, index, size);

	return createTensorObject(env, narrow);
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_select
  (JNIEnv * env, jobject t, jint dim, jint index){
	THTensor* tensor = getTensor(env, t);

	THTensor* select = THTensor_(newSelect)(
#ifdef CUDA
		state,
#endif
		tensor, dim, index);

	return createTensorObject(env, select);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_transpose
  (JNIEnv * env, jobject t, jobject target, jint d1, jint d2){
	THTensor* tensor = getTensor(env, t);

	THTensor* transpose = THTensor_(newTranspose)(
#ifdef CUDA
		state,
#endif
		tensor, d1, d2);

	return createTensorObject(env, transpose);
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_Tensor_diag
  (JNIEnv * env, jobject t, jobject target){
	THTensor* tensor = getTensor(env, t);

	THTensor* tensor2;
	if(target==NULL){
		tensor2 = THTensor_(new)(
#ifdef CUDA
			state
#endif
		);
	} else {
		tensor2 = getTensor(env, target);
	}

#ifdef CUDA
	// This is very inefficient, but diag is not critical anyway
	int size = tensor->size[0]; // should be square matrix, but is not checked here
	int i = 0;
	THTensor_(resize1d)(state, tensor2 , size);
	for(i  = 0; i< size; i++){
		float val = THTensor_(get2d)(state, tensor, i, i);
		THTensor_(set1d)(state, tensor2, i, val);
	}
#else
	THTensor_(diag)(tensor2, tensor, 0);
#endif
	return target == NULL ? createTensorObject(env, tensor2) : target;
}
