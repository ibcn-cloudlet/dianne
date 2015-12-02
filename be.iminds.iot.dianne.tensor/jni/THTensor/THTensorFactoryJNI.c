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

