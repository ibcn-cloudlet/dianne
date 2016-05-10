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
#include "be_iminds_iot_dianne_tensor_NativeTensorLoader.h"
#include "TensorLoader.h"

/** Exception handling from torch **/

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

/** Initialize and cleanup **/

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_NativeTensorLoader_init
  (JNIEnv * env, jobject loader){
	// cache class, field and method IDs for interacting with Tensor java object
	jclass tensorClass;
	char *className = "be/iminds/iot/dianne/tensor/Tensor";
	tensorClass = (*env)->FindClass(env, className);

	// jclass must be global reference
    TENSOR_CLASS = (*env)->NewGlobalRef(env, tensorClass);
	TENSOR_ADDRESS_FIELD = (*env)->GetFieldID(env, TENSOR_CLASS, "address", "J");
	TENSOR_INIT = (*env)->GetMethodID(env, TENSOR_CLASS, "<init>", "(J)V");


	// Set Torch error handler functions to throw Exceptions in Java
	(*env)->GetJavaVM(env, &jvm);
	THSetErrorHandler(torchErrorHandlerFunction, NULL);
	THSetArgErrorHandler(torchArgErrorHandlerFunction, NULL);

	// initialize CUDA
#ifdef CUDA
	state = (THCState*)malloc(sizeof(THCState));
	THCudaInit(state);
#endif
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_NativeTensorLoader_cleanup
  (JNIEnv * env, jobject loader){
	// release global reference to tensor class
	(*env)->DeleteGlobalRef(env, TENSOR_CLASS);

	// cleanup CUDA
#ifdef CUDA
	THCudaBlas_shutdown(state);
	free(state);
#endif
}


/** Tensor creation **/

THTensor* getTensor(JNIEnv* env, jobject o){
	if(o == NULL){
		// return new empty THTensor if jobject is null
		THTensor* t = THTensor_(new)(
#ifdef CUDA
				state
#endif
				);
		return t;
	}
	jlong address = (*env)->GetLongField(env, o, TENSOR_ADDRESS_FIELD);
	return (THTensor*) address;
}

THTensor* getTensor1d(JNIEnv* env, jobject o, int d0){
	THTensor* t = getTensor(env, o);

	THTensor_(resize1d)(
#ifdef CUDA
			state,
#endif
			t, d0);

	return t;
}

THTensor* getTensor2d(JNIEnv* env, jobject o, int d0, int d1){
	THTensor* t = getTensor(env, o);

	THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			t, d0, d1);

	return t;
}

THTensor* getTensor3d(JNIEnv* env, jobject o, int d0, int d1, int d2){
	THTensor* t = getTensor(env, o);

	THTensor_(resize3d)(
#ifdef CUDA
			state,
#endif
			t, d0, d1, d2);

	return t;
}

THTensor* getTensor4d(JNIEnv* env, jobject o, int d0, int d1, int d2, int d3){
	THTensor* t = getTensor(env, o);

	THTensor_(resize4d)(
#ifdef CUDA
			state,
#endif
			t, d0, d1, d2, d3);

	return t;
}

jobject createTensorObject(JNIEnv* env, THTensor* t){
	return (*env)->NewObject(env, TENSOR_CLASS, TENSOR_INIT, (jlong)t);
}
