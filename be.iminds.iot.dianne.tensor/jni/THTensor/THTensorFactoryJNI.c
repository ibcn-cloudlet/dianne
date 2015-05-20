#include "be_iminds_iot_dianne_tensor_impl_th_THTensorFactory.h"

#ifdef CUDA
#include "THCudaTensorJNI.h"
#else
#include "THTensorJNI.h"
#endif

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorFactory_init
  (JNIEnv * env, jobject o){


}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorFactory_cleanup
  (JNIEnv * env, jobject o){

}
