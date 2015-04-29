#include "be_iminds_iot_dianne_tensor_impl_th_THTensor.h"

#include "THTensorJNI.h"

THGenerator* generator = 0;

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_init
  (JNIEnv * env, jobject o, jfloatArray data, jintArray dims){
	THTensor * tensor;

	jsize noDims = (*env)->GetArrayLength(env, dims);

	jint *d = (*env)->GetIntArrayElements(env, dims, 0);
	if(noDims==1){
		tensor = THTensor_(newWithSize1d)(d[0]);
	} else if(noDims==2){
		tensor = THTensor_(newWithSize2d)(d[0], d[1]);
	} else if(noDims==3){
		tensor = THTensor_(newWithSize3d)(d[0], d[1], d[2]);
	} else if(noDims==4){
		tensor = THTensor_(newWithSize4d)(d[0], d[1], d[2], d[3]);
	} // for now only support up to 4D tensors...
	(*env)->ReleaseIntArrayElements(env, dims, d, 0);

	if(data != NULL){
		jsize len = (*env)->GetArrayLength(env, data);
		jfloat * floats = (*env)->GetFloatArrayElements(env, data, 0);

		jfloat* src_ptr = floats;
		real* dst_ptr = THTensor_(data)(tensor);
		int i =0;
		for(i=0;i<len;i++){
			*(dst_ptr++) = *(src_ptr++);
		}

		(*env)->ReleaseFloatArrayElements(env, data, floats, 0);
	}

	return (long) tensor;
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_free
  (JNIEnv * env, jobject o, jlong src){
	THTensor_(free)((THTensor *)src);
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
		THTensor_(resize1d)(tensor, index[0]);
	} else if(noDims==2){
		THTensor_(resize2d)(tensor, index[0], index[1]);
	} else if(noDims==3){
		THTensor_(resize3d)(tensor, index[0], index[1], index[2]);
	} else if(noDims==4){
		THTensor_(resize4d)(tensor, index[0], index[1], index[2], index[3]);
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
		val = THTensor_(get1d)(tensor, index[0]);
	} else if(noDims==2){
		val = THTensor_(get2d)(tensor, index[0], index[1]);
	} else if(noDims==3){
		val = THTensor_(get3d)(tensor, index[0], index[1], index[2]);
	} else if(noDims==4){
		val = THTensor_(get4d)(tensor, index[0], index[1], index[2], index[3]);
	} // for now only support up to 4D tensors...
	(*env)->ReleaseIntArrayElements(env, d, index, 0);

	return val;
}


JNIEXPORT jfloatArray JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_get__J
  (JNIEnv * env, jobject o, jlong src){
	THTensor* tensor = (THTensor *) src;
	real* ptr = THTensor_(data)(tensor);

	long size = tensor->storage->size;

	jfloatArray result;
	result = (*env)->NewFloatArray(env, size);
	if (result == NULL) {
	    return NULL;
	}
	int i;
	jfloat fill[size];
	for (i = 0; i < size; i++) {
	    fill[i] = *(ptr++);
	}
	(*env)->SetFloatArrayRegion(env, result, 0, size, fill);
	return result;
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_set__JF_3I
  (JNIEnv * env, jobject o, jlong src, jfloat val, jintArray d){
	THTensor* tensor = (THTensor*)src;

	jsize noDims = (*env)->GetArrayLength(env, d);

	jint *index = (*env)->GetIntArrayElements(env, d, 0);
	if(noDims==1){
		THTensor_(set1d)(tensor, index[0], val);
	} else if(noDims==2){
		THTensor_(set2d)(tensor, index[0], index[1], val);
	} else if(noDims==3){
		THTensor_(set3d)(tensor, index[0], index[1], index[2], val);
	} else if(noDims==4){
		THTensor_(set4d)(tensor, index[0], index[1], index[2], index[3], val);
	} // for now only support up to 4D tensors...
	(*env)->ReleaseIntArrayElements(env, d, index, 0);
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_set__J_3F
  (JNIEnv * env, jobject o, jlong src, jfloatArray data){
	THTensor* tensor = (THTensor *) src;
	jsize len = (*env)->GetArrayLength(env, data);
	jfloat * floats = (*env)->GetFloatArrayElements(env, data, 0);

	jfloat* src_ptr = floats;
	real* dst_ptr = THTensor_(data)(tensor);
	int i =0;
	for(i=0;i<len;i++){
		*(dst_ptr++) = *(src_ptr++);
	}

	(*env)->ReleaseFloatArrayElements(env, data, floats, 0);
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_fill
  (JNIEnv * env, jobject o, jlong src, jfloat val){
	THTensor* tensor = (THTensor*) src;
	THTensor_(fill)(src, val);
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_rand
  (JNIEnv * env, jobject o, jlong src){
	THTensor* tensor = (THTensor*) src;

	if(generator==0){
		generator = THGenerator_new();
	}

	THTensor_(uniform)(tensor, generator, 0, 1);
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_randn
  (JNIEnv * env, jobject o, jlong src){
	THTensor* tensor = (THTensor*) src;

	if(generator==0){
		generator = THGenerator_new();
	}

	THTensor_(normal)(tensor, random, 0, 1);
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_copyInto
  (JNIEnv * env, jobject o, jlong src, jlong res){
	THTensor* tensor = (THTensor*) src;
	THTensor* tensor2 = (THTensor*) res;

	real* src_ptr = THTensor_(data)(tensor);
	real* dst_ptr = THTensor_(data)(tensor2);
	int i = 0;
	for(i=0;i<tensor->storage->size;i++){
		*(dst_ptr++) = *(src_ptr++);
	}
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_narrow
  (JNIEnv * env, jobject o, jlong src, jint dim, jint index, jint size){
	THTensor* tensor = (THTensor*) src;
	THTensor* narrow = THTensor_(newNarrow)(tensor, dim, index, size);
	return narrow;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_select
  (JNIEnv * env, jobject o, jlong src, jint dim, jint index){
	THTensor* tensor = (THTensor*) src;
	THTensor* select = THTensor_(newSelect)(tensor, dim, index);
	return select;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_transpose
  (JNIEnv * env, jobject o, jlong src, jint d1, jint d2){
	THTensor* tensor = (THTensor*) src;
	return THTensor_(newTranspose)(tensor, d1, d2);
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_diag
  (JNIEnv * env, jobject o, jlong src, jlong dst){
	THTensor* tensor = (THTensor*) src;
	THTensor* tensor2;
	if(dst==0){
		tensor2 = THTensor_(new)();
	} else {
		tensor2 = (THTensor*) dst;
	}
	THTensor_(diag)(tensor2, tensor, 0);
	return tensor2;
}


JNIEXPORT jboolean JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_equals
  (JNIEnv * env, jobject o, jlong src, jlong other){
	THTensor* tensor = (THTensor*) src;
	THTensor* tensor2 = (THTensor*) other;

	THByteTensor* neq = THByteTensor_new();
	THTensor_(neTensor)(neq, tensor, tensor2);
	accreal sum = THByteTensor_sumall(neq);
	THByteTensor_free(neq);

	if(sum==0){
		return 1;
	} else {
		return 0;
	}
}
