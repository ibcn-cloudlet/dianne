#include "be_iminds_iot_dianne_tensor_NativeTensorLoader.h"
#include "TensorLoader.h"

#include "CudnnTensor.h"

#include <string.h>

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

	if(workspaceSize > 0){
		THCudaCheck(cudaFree(workspace));
	}
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_NativeTensorLoader_option
  (JNIEnv * env, jobject loader, jstring keyString, jstring valueString){

	const char* key = env->GetStringUTFChars(keyString, 0);
	const char* value = env->GetStringUTFChars(valueString, 0);

	if(strcmp(key, "shareWorkspace")==0){
		if(strcmp(value, "true")==0){
			shareWorkspace = 1;
		} else {
			shareWorkspace = 0;
		}
		printf("Configure %s %d \n", key, shareWorkspace);

	} else if(strcmp(key, "workspaceLimit")==0){
		workspaceLimit = atoi(value);

		printf("Configure %s %d \n", key, workspaceLimit);
	} else if(strcmp(key, "convFwAlg")==0){
		convFwAlg = atoi(value);

		printf("Configure %s %d \n", key, convFwAlg);
	} else if(strcmp(key, "convBwAlg")==0){
		convBwAlg = atoi(value);

		printf("Configure %s %d \n", key, convBwAlg);
	} else if(strcmp(key, "convAgAlg")==0){
		convAgAlg = atoi(value);

		printf("Configure %s %d \n", key, convAgAlg);
	}

	env->ReleaseStringUTFChars(keyString, key);
	env->ReleaseStringUTFChars(valueString, value);

}
