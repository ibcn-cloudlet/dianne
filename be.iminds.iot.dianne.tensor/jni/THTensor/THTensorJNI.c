#include "be_iminds_iot_dianne_tensor_impl_th_THTensor.h"

#include "THTensorJNI.h"

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
		memcpy( floats, THTensor_(data)(tensor) , len * sizeof(float) );
		(*env)->ReleaseFloatArrayElements(env, data, floats, 0);
	}

	return (long) tensor;
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_free
  (JNIEnv * env, jobject o, jlong ptr){
	THTensor_(free)((THTensor *)ptr);
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_reshape
  (JNIEnv * env, jobject o, jlong src, jintArray dims){

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


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_get__J
  (JNIEnv * env, jobject o, jlong src){

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

}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_fill
  (JNIEnv * env, jobject o, jlong src, jfloat val){

}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_rand
  (JNIEnv * env, jobject o, jlong src){

}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_randn
  (JNIEnv * env, jobject o, jlong src){

}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_copyInto
  (JNIEnv * env, jobject o, jlong src, jlong res){

}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_narrow
  (JNIEnv * env, jobject o, jlong src, jint dim, jint index, jint size){

}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_select
  (JNIEnv * env, jobject o, jlong src, jint dim, jint index){

}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_transpose
  (JNIEnv * env, jobject o, jlong src, jlong res, jint d1, jint d2){

}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_diag
  (JNIEnv * env, jobject o, jlong src, jlong res){

}


JNIEXPORT jboolean JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_equals
  (JNIEnv * env, jobject o, jlong src, jlong other){

}
