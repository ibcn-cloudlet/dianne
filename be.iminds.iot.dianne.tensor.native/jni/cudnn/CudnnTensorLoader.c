#include "be_iminds_iot_dianne_tensor_NativeTensorLoader.h"
#include "TensorLoader.h"

#include "CudnnTensor.h"

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_NativeTensorLoader_init
  (JNIEnv * env, jobject loader, jint device){
	// initialize TH Tensors
	initTH(env, device);

	// init cudnn
	checkCUDNN(cudnnCreate(&cudnnHandle));
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_NativeTensorLoader_cleanup
  (JNIEnv * env, jobject loader){
	// cleanup TH
	cleanupTH(env);

	// cleanup cudnn
	checkCUDNN(cudnnDestroy(cudnnHandle));

}
