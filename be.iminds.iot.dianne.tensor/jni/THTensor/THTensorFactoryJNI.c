#include "be_iminds_iot_dianne_tensor_impl_th_THTensorFactory.h"

#ifdef CUDA
#include "THCudaTensorJNI.h"
#else
#include "THTensorJNI.h"
#endif

JavaVM* jvm;

static void throwException(const char * msg){
	JNIEnv* env;
	(*jvm)->AttachCurrentThread(jvm, (void**)&env, NULL);

	jclass exClass;
	char *className = "java/lang/Exception";

	exClass = (*env)->FindClass( env, className);

	(*env)->ThrowNew( env, exClass, msg );
}

static void torchErrorHandlerFunction(const char *msg, void *data){
	throwException(msg);
}

static void torchArgErrorHandlerFunction(int argNumber, const char *msg, void *data){
	throwException(msg);
}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorFactory_init
  (JNIEnv * env, jobject o){

#ifdef CUDA
	state = (THCState*)malloc(sizeof(THCState));
	THCudaInit(state);
#endif

	// Set other error handler functions to throw Exceptions in Java
	(*env)->GetJavaVM(env, &jvm);
	THSetErrorHandler(torchErrorHandlerFunction, NULL);
	THSetArgErrorHandler(torchArgErrorHandlerFunction, NULL);

}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorFactory_cleanup
  (JNIEnv * env, jobject o){
#ifdef CUDA
	THCudaBlas_shutdown(state);
	free(state);
#endif
}

