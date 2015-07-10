#include "be_iminds_iot_dianne_tensor_impl_th_THTensor.h"

#ifdef CUDA
#include "THCudaTensorJNI.h"
#else
#include "THTensorJNI.h"
#endif

THGenerator* generator = 0;

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_init
  (JNIEnv * env, jobject o, jfloatArray data, jintArray dims){
	THTensor * tensor;

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


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_free
  (JNIEnv * env, jobject o, jlong src){
	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			(THTensor *)src);
}

JNIEXPORT jintArray JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_dims
  (JNIEnv * env, jobject o, jlong src){
	THTensor* tensor = (THTensor*)src;
	long* ptr = tensor->size;

	int size = tensor->nDimension;
	jintArray result;
	result = (*env)->NewIntArray(env, size);
	if (result == NULL) {
	    return NULL;
	}
	int i;
	jint fill[size];
	for (i = 0; i < size; i++) {
	    fill[i] = *(ptr++);
	}
	(*env)->SetIntArrayRegion(env, result, 0, size, fill);
	return result;
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_reshape
  (JNIEnv * env, jobject o, jlong src, jintArray dims){
	THTensor* tensor = (THTensor*)src;

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


JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_get__J_3I
  (JNIEnv * env, jobject o, jlong src, jintArray d){
	float val;
	THTensor* tensor = (THTensor*)src;

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


JNIEXPORT jfloatArray JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_get__J
  (JNIEnv * env, jobject o, jlong src){
	THTensor* tensor = (THTensor *) src;
	real* ptr = THTensor_(data)(
#ifdef CUDA
			state,
#endif
			tensor);

	real* data = ptr;

#ifdef CUDA
	long bufferSize = tensor->storage->size*sizeof(real);
	real* buffer = malloc(bufferSize);
	cudaMemcpy(buffer, ptr, bufferSize, cudaMemcpyDeviceToHost);
	(*env)->SetFloatArrayRegion(env, result, 0, size, buffer);
	data = buffer;
# endif

	// calculate actual size
	// (can be different of underlying data array in case of narrowed tensor)
	int size = 1;
	int i;
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


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_set__JF_3I
  (JNIEnv * env, jobject o, jlong src, jfloat val, jintArray d){
	THTensor* tensor = (THTensor*)src;

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


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_set__J_3F
  (JNIEnv * env, jobject o, jlong src, jfloatArray data){
	THTensor* tensor = (THTensor *) src;
	jsize len = (*env)->GetArrayLength(env, data);
	jfloat * floats = (*env)->GetFloatArrayElements(env, data, 0);

#ifdef CUDA
	cudaMemcpy(THTensor_(data)(state, tensor), floats, len*sizeof(real), cudaMemcpyHostToDevice);
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


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_fill
  (JNIEnv * env, jobject o, jlong src, jfloat val){
	THTensor* tensor = (THTensor*) src;
	THTensor_(fill)(
#ifdef CUDA
			state,
#endif
			src, val);
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_rand
  (JNIEnv * env, jobject o, jlong src){
	THTensor* tensor = (THTensor*) src;

#ifdef CUDA
	THTensor_(uniform)(state, tensor, 0, 1);
#else
	if(generator==0){
		generator = THGenerator_new();
	}

	THTensor_(uniform)(tensor, generator, 0, 1);
#endif
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_randn
  (JNIEnv * env, jobject o, jlong src){
	THTensor* tensor = (THTensor*) src;

#ifdef CUDA
	THTensor_(normal)(state, tensor, 0, 1);
#else
	if(generator==0){
		generator = THGenerator_new();
	}

	THTensor_(normal)(tensor, random, 0, 1);
#endif
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_copyInto
  (JNIEnv * env, jobject o, jlong src, jlong res){
	THTensor* tensor = (THTensor*) src;
	THTensor* tensor2 = (THTensor*) res;

#ifdef CUDA
	THTensor_(copyCuda)(state, tensor2, tensor);
#else
	THTensor_(copyFloat)(tensor2, tensor);
#endif
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_narrow
  (JNIEnv * env, jobject o, jlong src, jint dim, jint index, jint size){
	THTensor* tensor = (THTensor*) src;
	THTensor* narrow = THTensor_(newNarrow)(
#ifdef CUDA
			state,
#endif
			tensor, dim, index, size);
	return narrow;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_select
  (JNIEnv * env, jobject o, jlong src, jint dim, jint index){
	THTensor* tensor = (THTensor*) src;
	THTensor* select = THTensor_(newSelect)(
#ifdef CUDA
			state,
#endif
			tensor, dim, index);
	return select;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_transpose
  (JNIEnv * env, jobject o, jlong src, jint d1, jint d2){
	THTensor* tensor = (THTensor*) src;
	return THTensor_(newTranspose)(
#ifdef CUDA
			state,
#endif
			tensor, d1, d2);
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_diag
  (JNIEnv * env, jobject o, jlong src, jlong dst){
	THTensor* tensor = (THTensor*) src;
	THTensor* tensor2;
	if(dst==0){
		tensor2 = THTensor_(new)(
#ifdef CUDA
			state
#endif
		);
	} else {
		tensor2 = (THTensor*) dst;
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
	return tensor2;
}


JNIEXPORT jboolean JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_equals
  (JNIEnv * env, jobject o, jlong src, jlong other){
	THTensor* tensor = (THTensor*) src;
	THTensor* tensor2 = (THTensor*) other;

	accreal sum;
#ifdef CUDA
	THCudaTensor* neq = THCudaTensor_new(state);
	THTensor_(neTensor)(state, neq, tensor, tensor2);
	sum = THCudaTensor_sumall(state, neq);
	THCudaTensor_free(state, neq);
#else
	THByteTensor* neq = THByteTensor_new();
	THTensor_(neTensor)(neq, tensor, tensor2);
	sum = THByteTensor_sumall(neq);
	THByteTensor_free(neq);
#endif

	if(sum==0){
		return 1;
	} else {
		return 0;
	}
}
