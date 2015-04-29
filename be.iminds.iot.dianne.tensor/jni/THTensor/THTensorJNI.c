#include "be_iminds_iot_dianne_tensor_impl_th_THTensor.h"

#include "THTensorJNI.h"

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_init
  (JNIEnv * env, jobject o, jfloatArray data, jintArray dims){
	THTensor * tensor;

	jsize len = (*env)->GetArrayLength(env, dims);
	printf("CREATE %d-D TENSOR \n", len);

	jint *d = (*env)->GetIntArrayElements(env, dims, 0);
	if(len==1){
		tensor = THTensor_(newWithSize1d)(d[0]);
	} else if(len==2){
		tensor = THTensor_(newWithSize2d)(d[0], d[1]);
	} else if(len==3){
		tensor = THTensor_(newWithSize3d)(d[0], d[1], d[2]);
	} else if(len==4){
		tensor = THTensor_(newWithSize4d)(d[0], d[1], d[2], d[3]);
	} // for now only support up to 4D tensors...
	(*env)->ReleaseIntArrayElements(env, dims, d, 0);

	return (long) tensor;
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensor_free
  (JNIEnv * env, jobject o, jlong ptr){
	printf("FREE TH TENSOR \n");

	THTensor_(free)((THTensor *)ptr);
}
