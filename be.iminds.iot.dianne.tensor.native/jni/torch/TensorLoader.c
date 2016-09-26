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
int CURRENT_GPU;
static jfieldID TENSOR_ADDRESS_FIELD;
static jmethodID TENSOR_INIT;
static jclass TENSOR_CLASS;
static jmethodID SYSTEM_GC;
static jclass SYSTEM_CLASS;
static jclass EXCEPTION_CLASS;

void throwException(const char * msg){
	JNIEnv* env;
	jvm->AttachCurrentThread((void**)&env, NULL);
	env->ThrowNew(EXCEPTION_CLASS, msg );
	jvm->DetachCurrentThread();
}

static void torchErrorHandlerFunction(const char *msg, void *data){
	throwException(msg);
}

static void torchArgErrorHandlerFunction(int argNumber, const char *msg, void *data){
	throwException(msg);
}


static void gcFunction(void *data){
	JNIEnv* env;
	jvm->AttachCurrentThread((void**)&env, NULL);
	env->CallStaticVoidMethod(SYSTEM_CLASS, SYSTEM_GC);
	jvm->DetachCurrentThread();
}


void initTH(JNIEnv* env, int device){
	// cache class, field and method IDs for interacting with Tensor java object
	jclass tensorClass;
	char *className = (char*)"be/iminds/iot/dianne/tensor/Tensor";
	tensorClass = env->FindClass(className);

	// jclass must be global reference
    TENSOR_CLASS = (jclass) env->NewGlobalRef(tensorClass);
	TENSOR_ADDRESS_FIELD = env->GetFieldID(TENSOR_CLASS, "address", "J");
	TENSOR_INIT = env->GetMethodID(TENSOR_CLASS, "<init>", "(J)V");

	jclass systemClass;
	char *systemClassName = (char*)"java/lang/System";
	systemClass = env->FindClass(systemClassName);
    SYSTEM_CLASS = (jclass) env->NewGlobalRef(systemClass);
	SYSTEM_GC = env->GetStaticMethodID(SYSTEM_CLASS, "gc", "()V");

	jclass exceptionClass;
	char *exClassName = (char*)"java/lang/Exception";
	exceptionClass = env->FindClass(exClassName);
    EXCEPTION_CLASS = (jclass) env->NewGlobalRef(exceptionClass);

	// Set Torch error handler functions to throw Exceptions in Java
	env->GetJavaVM(&jvm);
	THSetErrorHandler(torchErrorHandlerFunction, NULL);
	THSetArgErrorHandler(torchArgErrorHandlerFunction, NULL);

	// Set Torch GC handler
	THSetGCHandler(gcFunction, NULL);

	// initialize CUDA
#ifdef CUDA
	if(state == 0){
		state = (THCState*)malloc(sizeof(THCState));
		THCudaInit(state);

		if(device >= 0){
			CURRENT_GPU = device;
			selectGPU(device);
		} else {
			CURRENT_GPU = 0;
		}
	}
	THCudaCheck(cudaGetLastError());
#endif

}

void cleanupTH(JNIEnv* env){
	// release global class references
	env->DeleteGlobalRef(TENSOR_CLASS);
	env->DeleteGlobalRef(SYSTEM_CLASS);
	env->DeleteGlobalRef(EXCEPTION_CLASS);


	// cleanup CUDA
#ifdef CUDA
	THCudaShutdown(state);
	free(state);
#endif
}


/** Initialize and cleanup **/

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_NativeTensorLoader_init
  (JNIEnv * env, jobject loader, jint device){
	initTH(env, device);
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_NativeTensorLoader_cleanup
  (JNIEnv * env, jobject loader){
	cleanupTH(env);
}


/** Tensor creation **/

THTensor* getTensor(JNIEnv* env, jobject o){
#ifdef CUDA
	selectGPU(CURRENT_GPU);
#endif

	if(o == NULL){

		// return new empty THTensor if jobject is null
		THTensor* t = THTensor_(new)(
#ifdef CUDA
				state
#endif
				);
		return t;
	}
	jlong address = env->GetLongField(o, TENSOR_ADDRESS_FIELD);
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

THTensor* getTensor5d(JNIEnv* env, jobject o, int d0, int d1, int d2, int d3, int d4){
	THTensor* t = getTensor(env, o);

	THTensor_(resize5d)(
#ifdef CUDA
			state,
#endif
			t, d0, d1, d2, d3, d4);

	return t;
}

jobject createTensorObject(JNIEnv* env, THTensor* t){
	return env->NewObject(TENSOR_CLASS, TENSOR_INIT, (jlong)t);
}


#ifdef CUDA
void selectGPU(int d){
	int current;

	THCudaCheck(cudaGetDevice(&current));
	if(current == d)
		return;

	THCudaCheck(cudaSetDevice(d));	
	THCRandom_setGenerator(state, d);

	/* The stream is per device, so update the stream as well */
	THCState_setStream(state, d, THCState_getCurrentStreamIndex(state));
	THCState_setBlasHandle(state, d, THCState_getCurrentBlasHandleIndex(state));
}
#endif
