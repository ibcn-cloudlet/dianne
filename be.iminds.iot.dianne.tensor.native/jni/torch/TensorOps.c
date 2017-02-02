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
#include "be_iminds_iot_dianne_tensor_TensorOps.h"
#include "TensorLoader.h"

#ifdef CUDA
#include "CudaTensor.h"
#include "THCudaTensorOps.h"
#else
#include "Tensor.h"
#endif

// convert tensor to 1d vector
THTensor* getVector(THTensor* l){
	THTensor* v = THTensor_(newContiguous)(
#ifdef CUDA
			state,
#endif
			l);

	int size = THTensor_(nElement)(
#ifdef CUDA
			state,
#endif
			v);

	THTensor_(resize1d)(
#ifdef CUDA
			state,
#endif
			v , size);
	return v;
}

void releaseVector(THTensor* v){
	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			v);
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_add__Lbe_iminds_iot_dianne_tensor_Tensor_2Lbe_iminds_iot_dianne_tensor_Tensor_2F
  (JNIEnv * env, jclass c, jobject res, jobject tensor, jfloat val){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(add)(
#ifdef CUDA
			state,
#endif
			r, t, val);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_add__Lbe_iminds_iot_dianne_tensor_Tensor_2Lbe_iminds_iot_dianne_tensor_Tensor_2Lbe_iminds_iot_dianne_tensor_Tensor_2
  (JNIEnv * env, jclass c, jobject res, jobject tensor1, jobject tensor2){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor1);
	THTensor* a = getTensor(env, tensor2);
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			r, t, 1, a);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_add__Lbe_iminds_iot_dianne_tensor_Tensor_2Lbe_iminds_iot_dianne_tensor_Tensor_2FLbe_iminds_iot_dianne_tensor_Tensor_2
  (JNIEnv * env, jclass c, jobject res, jobject tensor1, jfloat val, jobject tensor2){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor1);
	THTensor* a = getTensor(env, tensor2);
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			r, t, val, a);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_sub__Lbe_iminds_iot_dianne_tensor_Tensor_2Lbe_iminds_iot_dianne_tensor_Tensor_2F
  (JNIEnv * env, jclass c, jobject res, jobject tensor, jfloat val){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(add)(
#ifdef CUDA
			state,
#endif
			r, t, -val);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_sub__Lbe_iminds_iot_dianne_tensor_Tensor_2Lbe_iminds_iot_dianne_tensor_Tensor_2Lbe_iminds_iot_dianne_tensor_Tensor_2
  (JNIEnv * env, jclass c, jobject res, jobject tensor1, jobject tensor2){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor1);
	THTensor* s = getTensor(env, tensor2);
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			r, t, -1, s);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_sub__Lbe_iminds_iot_dianne_tensor_Tensor_2Lbe_iminds_iot_dianne_tensor_Tensor_2FLbe_iminds_iot_dianne_tensor_Tensor_2
  (JNIEnv * env, jclass c, jobject res, jobject tensor1, jfloat val, jobject tensor2){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor1);
	THTensor* s = getTensor(env, tensor2);
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			r, t, -val, s);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_mul
  (JNIEnv * env, jclass c, jobject res, jobject tensor, jfloat val){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(mul)(
#ifdef CUDA
			state,
#endif
			r, t, val);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_cmul
  (JNIEnv * env, jclass c, jobject res, jobject tensor1, jobject tensor2){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor1);
	THTensor* m = getTensor(env, tensor2);
	THTensor_(cmul)(
#ifdef CUDA
			state,
#endif
			r, t, m);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_addcmul
  (JNIEnv * env, jclass c, jobject res, jobject tensor1, jfloat val, jobject tensor2, jobject tensor3){
	THTensor* r = getTensor(env, res);
	THTensor* t1 = getTensor(env, tensor1);
	THTensor* t2 = getTensor(env, tensor2);
	THTensor* t3 = getTensor(env, tensor3);
	THTensor_(addcmul)(
#ifdef CUDA
			state,
#endif
			r, t1, val, t2, t3);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_div
  (JNIEnv * env, jclass c, jobject res, jobject tensor, jfloat val){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(div)(
#ifdef CUDA
			state,
#endif
			r, t, val);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_cdiv
  (JNIEnv * env, jclass c, jobject res, jobject tensor1, jobject tensor2){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor1);
	THTensor* d = getTensor(env, tensor2);
	THTensor_(cdiv)(
#ifdef CUDA
			state,
#endif
			r, t, d);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_addcdiv
  (JNIEnv * env, jclass c, jobject res, jobject tensor1, jfloat val, jobject tensor2, jobject tensor3){
	THTensor* r = getTensor(env, res);
	THTensor* t1 = getTensor(env, tensor1);
	THTensor* t2 = getTensor(env, tensor2);
	THTensor* t3 = getTensor(env, tensor3);
	THTensor_(addcdiv)(
#ifdef CUDA
			state,
#endif
			r, t1, val, t2, t3);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_dot
  (JNIEnv * env, jclass c, jobject tensor1, jobject tensor2){
	THTensor* vec1 = getTensor(env, tensor1);
	THTensor* vec2 = getTensor(env, tensor2);
	return THTensor_(dot)(
#ifdef CUDA
			state,
#endif
			vec1, vec2);
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_vv
  (JNIEnv * env, jclass c, jobject res, jobject v1, jobject v2){
	THTensor* vec1= getVector(getTensor(env, v1));
	THTensor* vec2 = getVector(getTensor(env, v2));

	THTensor* r = getTensor2d(env, res, vec1->size[0], vec2->size[0]);

	THTensor_(addr)(
#ifdef CUDA
			state,
#endif
			r, 0.0f, r, 1.0f, vec1, vec2);

	releaseVector(vec1);
	releaseVector(vec2);

	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_mv
  (JNIEnv * env, jclass c, jobject res, jobject m, jobject v){
	THTensor* mat = getTensor(env, m);
	THTensor* vec = getVector(getTensor(env, v));

	THTensor* r = getTensor1d(env, res, mat->size[0]);
	THTensor* rv = getVector(r);

	THTensor_(addmv)(
#ifdef CUDA
			state,
#endif
			rv, 0.0f, rv, 1.0f, mat, vec);

	releaseVector(vec);
	releaseVector(rv);

	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_tmv
  (JNIEnv * env, jclass c, jobject res, jobject m, jobject v){
	THTensor* mat = getTensor(env, m);
	THTensor* vec = getVector(getTensor(env, v));
	THTensor* r = getTensor1d(env, res, mat->size[1]);
	THTensor* rv = getVector(r);

	THTensor* transpose = THTensor_(newTranspose)(
#ifdef CUDA
			state,
#endif
			mat, 0, 1);

	THTensor_(addmv)(
#ifdef CUDA
			state,
#endif
			rv, 0.0f, rv, 1.0f, transpose, vec);

	releaseVector(vec);
	releaseVector(rv);

	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			transpose);

	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_mm
  (JNIEnv * env, jclass c, jobject res, jobject m1, jobject m2){
	THTensor* mat1 = getTensor(env, m1);
	THTensor* mat2 = getTensor(env, m2);
	THTensor* r = getTensor2d(env, res, mat1->size[0], mat2->size[1]);

	THTensor_(addmm)(
#ifdef CUDA
			state,
#endif
			r, 0.0f, r, 1.0f, mat1, mat2);

	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_tmm
  (JNIEnv * env, jclass c, jobject res, jobject m1, jobject m2){
	THTensor* mat1 = getTensor(env, m1);
	THTensor* mat2 = getTensor(env, m2);
	THTensor* r = getTensor2d(env, res, mat1->size[1], mat2->size[1]);

	THTensor* transpose = THTensor_(newTranspose)(
#ifdef CUDA
			state,
#endif
			mat1, 0, 1);

	THTensor_(addmm)(
#ifdef CUDA
			state,
#endif
			r, 0.0f, r, 1.0f, transpose, mat2);

	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			transpose);

	return res == NULL ? createTensorObject(env, r) : res;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_addvv
  (JNIEnv * env, jclass c, jobject res, jobject m, jobject v1, jobject v2){
	THTensor* mat = getTensor(env, m);
	THTensor* vec1 = getVector(getTensor(env, v1));
	THTensor* vec2 = getVector(getTensor(env, v2));
	THTensor* r = getTensor2d(env, res, mat->size[0], mat->size[1]);

	THTensor_(addr)(
#ifdef CUDA
			state,
#endif
			r, 1.0f, mat, 1.0f, vec1, vec2);

	releaseVector(vec1);
	releaseVector(vec2);

	return res == NULL ? createTensorObject(env, r) : res;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_addmv
  (JNIEnv * env, jclass c, jobject res, jobject v1, jobject m, jobject v2){
	THTensor* mat = getTensor(env, m);
	THTensor* vec1 = getVector(getTensor(env, v1));
	THTensor* vec2 = getVector(getTensor(env, v2));
	THTensor* r = getTensor1d(env, res, vec1->size[0]);
	THTensor* rv = getVector(r);

	THTensor_(addmv)(
#ifdef CUDA
			state,
#endif
			rv, 1.0f, vec1, 1.0f, mat, vec2);

	releaseVector(vec1);
	releaseVector(vec2);
	releaseVector(rv);

	return res == NULL ? createTensorObject(env, r) : res;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_addmm
  (JNIEnv * env, jclass c, jobject res, jobject m1, jobject m2, jobject m3){
	THTensor* mat1 = getTensor(env, m1);
	THTensor* mat2 = getTensor(env, m2);
	THTensor* mat3 = getTensor(env, m3);
	THTensor* r = getTensor2d(env, res, mat3->size[0], mat3->size[1]);

	THTensor_(addmm)(
#ifdef CUDA
			state,
#endif
			r, 1.0f, mat1, 1.0f, mat2, mat3);

	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_pow
  (JNIEnv * env, jclass c, jobject res, jobject tensor, jfloat val){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);
	THTensor_(pow)(
#ifdef CUDA
			state,
#endif
			r, t, val);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_exp
  (JNIEnv * env, jclass c, jobject res, jobject tensor){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);
	THTensor_(exp)(
#ifdef CUDA
			state,
#endif
			r, t);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_log
  (JNIEnv * env, jclass c, jobject res, jobject tensor){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);
	THTensor_(log)(
#ifdef CUDA
			state,
#endif
			r, t);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_sqrt
  (JNIEnv * env, jclass c, jobject res, jobject tensor){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

	THTensor_(sqrt)(
#ifdef CUDA
			state,
#endif
			r, t);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_abs
  (JNIEnv * env, jclass c, jobject res, jobject tensor){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

	THTensor_(abs)(
#ifdef CUDA
			state,
#endif
			r, t);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_sign
  (JNIEnv * env, jclass c, jobject res, jobject tensor){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

	THTensor_(sign)(
#ifdef CUDA
			state,
#endif
			r, t);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_round
  (JNIEnv * env, jclass c, jobject res, jobject tensor){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

	THTensor_(round)(
#ifdef CUDA
			state,
#endif
			r, t);
	return res == NULL ? createTensorObject(env, r) : res;
}



JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_sum
  (JNIEnv * env, jclass c, jobject tensor){
	THTensor* t = getTensor(env, tensor);
	return THTensor_(sumall)(
#ifdef CUDA
			state,
#endif
			t);
}



JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_max
  (JNIEnv * env, jclass c, jobject tensor){
	THTensor* t = getTensor(env, tensor);
	return THTensor_(maxall)(
#ifdef CUDA
			state,
#endif
			t);
}



JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_min
  (JNIEnv * env, jclass c, jobject tensor){
	THTensor* t = getTensor(env, tensor);
	return THTensor_(minall)(
#ifdef CUDA
			state,
#endif
			t);
}



JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_mean
  (JNIEnv * env, jclass c, jobject tensor){
	THTensor* t = getTensor(env, tensor);
	return THTensor_(meanall)(
#ifdef CUDA
			state,
#endif
			t);
}



JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_argmax
  (JNIEnv * env, jclass c, jobject tensor){
	THTensor* t = getTensor(env, tensor);
	THTensor* vec = getVector(t);

	int index = 0;

#ifdef CUDA
	index = THCudaTensor_argmax(state, vec);
#else
	real* data_ptr = THTensor_(data)(vec);
	real max = data_ptr[0];
	long size = vec->size[0];
	long stride = vec->stride[0];
	int i = 0;
	for(i=0;i<size;i++){
		if(max < *data_ptr){
			max = *data_ptr;
			index = i;
		}
		data_ptr += stride;
	}
#endif
	releaseVector(vec);
	return index;
}



JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_argmin
  (JNIEnv * env, jclass c, jobject tensor){
	THTensor* t = getTensor(env, tensor);
	THTensor* vec = getVector(t);

	int index = 0;

#ifdef CUDA
	index = THCudaTensor_argmin(state, vec);
#else
	real* data_ptr = THTensor_(data)(vec);
	real min = data_ptr[0];
	long size = vec->size[0];
	long stride = vec->stride[0];
	int i = 0;
	for(i=0;i<size;i++){
		if(min > *data_ptr){
			min = *data_ptr;
			index = i;
		}
		data_ptr += stride;
	}
#endif
	releaseVector(vec);
	return index;
}

#ifndef CUDA
// helper function on CPU only
void scale2d(THTensor* r, THTensor* t){
	int y_in = t->size[t->nDimension-2];
	int x_in = t->size[t->nDimension-1];

	int y_out = r->size[r->nDimension-2];
	int x_out = r->size[r->nDimension-1];

	float s_y = (y_in-1)/(float)(y_out-1);
	float s_x = (x_in-1)/(float)(x_out-1);

	int channels = r->size[r->nDimension-3];

	real* src_ptr = THTensor_(data)(t);
	real* dst_ptr = THTensor_(data)(r);

	int c,y,x,cc;
	float xx,yy,dx,dy;
	real v1,v2,v3,v4,v;
	int x1,x2,y1,y2;

	// strides depend on input dimension
	int stride_c = t->nDimension == 3 ? t->stride[0] : 0;
	int stride_y = t->nDimension == 3 ? t->stride[1] : t->stride[0];
	int stride_x = t->nDimension == 3 ? t->stride[2] : t->stride[1];

	for(c=0;c<channels;c++){
		for(y=0;y<y_out;y++){
			for(x=0;x<x_out;x++){
				cc = c;
				if(t->nDimension == 2 || cc > t->size[0]){
					cc = 1;
				}

				yy = y*s_y;
				xx = x*s_x;

				// bilinear interpolation
				x1 = (int)xx;
				x2 = x1+1;
				if(x2==x_in)
					x2--;
				y1 = (int)yy;
				y2 = y1+1;
				if(y2==y_in)
					y2--;

				v1 = src_ptr[cc*stride_c + y1*stride_y + x1*stride_x];
				v2 = src_ptr[cc*stride_c + y1*stride_y + x2*stride_x];
				v3 = src_ptr[cc*stride_c + y2*stride_y + x1*stride_x];
				v4 = src_ptr[cc*stride_c + y2*stride_y + x2*stride_x];

				dx = xx-x1;
				dy = yy-y1;

				v = v1*(1-dy)*(1-dx)
						 + v2 * (1-dy)*(dx)
						 + v3 * (dy)*(1-dx)
						 + v4 * (dx)*(dy);

				dst_ptr[x_out*y_out*c + x_out*y + x] = v;
			}
		}
	}
}
#endif

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_scale2D
  (JNIEnv * env, jclass clazz, jobject res, jobject tensor, jintArray dims){
	THTensor* t = getTensor(env, tensor);
	THTensor* r;

	jsize noDims = env->GetArrayLength(dims);
	jint *d = env->GetIntArrayElements(dims, 0);

	if(noDims==2){
		r = getTensor3d(env, res, 1, d[0], d[1]);
	} else {
		r = getTensor3d(env, res, d[0], d[1], d[2]);
	}

#ifdef CUDA
	THCudaTensor_scale2d(state, r, t);
#else
	scale2d(r, t);
#endif

	if(noDims==2){
		THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			r, d[0], d[1]);
	}

	env->ReleaseIntArrayElements(dims, d, 0);

	return res == NULL ? createTensorObject(env, r) : res;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_frame
  (JNIEnv * env, jclass c, jobject res, jobject tensor, jintArray dims){
	THTensor* t = getTensor(env, tensor);
	THTensor* r;

	jsize noDims = env->GetArrayLength(dims);
	jint *d = env->GetIntArrayElements(dims, 0);

	int noBatches;
	if(t->nDimension == noDims + 1){
		noBatches = t->size[0];
	} else {
		noBatches = 1;
	}

	// make sure input and output or 4d  [batch, channel, y ,x]
	if(noDims==2){
		if(noBatches > 1)
			r = getTensor4d(env, res, noBatches, 1, d[0], d[1]);
		else
			r = getTensor4d(env, res, 1, 1, d[0], d[1]);
	} else {
		if(noBatches > 1)
			r = getTensor4d(env, res, noBatches, d[0], d[1], d[2]);
		else
			r = getTensor4d(env, res, 1, d[0], d[1], d[2]);
	}

	// check input size
	int i;
	int correct = 1;
	for(i = 0;i<noDims;i++){
		if(t->size[noBatches == 1 ? i : i+1] != d[i])
			correct = 0;
	}

	if(correct){
		// input has correct size, just copy to output
#ifdef CUDA
		THTensor_(copyCuda)(state, r, t);
#else
		THTensor_(copyFloat)(r, t);
#endif
	} else {
		// narrow and scale input
		if(t->nDimension == 2){
			THTensor_(resize4d)(
#ifdef CUDA
					state,
#endif
					t, 1, 1, t->size[0], t->size[1]);
		} else if(t->nDimension == 3){
			if(noBatches == 1){
				THTensor_(resize4d)(
#ifdef CUDA
						state,
#endif
						t, 1, t->size[0], t->size[1], t->size[2]);
			} else {
				THTensor_(resize4d)(
#ifdef CUDA
					state,
#endif
					t, t->size[0], 1, t->size[1], t->size[2]);
			}
		} else {
			THTensor_(resize4d)(
#ifdef CUDA
				state,
#endif
				t, t->size[0], t->size[1], t->size[2], t->size[3]);
		}

		// narrow
		float sx = (float)(t->size[3])/r->size[3];
		float sy = (float)(t->size[2])/r->size[2];

		float s = sx < sy ? sx : sy;

		// dim y
		int size = (int)(r->size[2]*s);
		int index = (t->size[2]-size)/2;

		THTensor* narrow = THTensor_(newNarrow)(
#ifdef CUDA
			state,
#endif
			t, 2, index, size);

		// dim x
		size = (int)(r->size[3]*s);
		index = (t->size[3]-size)/2;

		THTensor_(narrow)(
#ifdef CUDA
			state,
#endif
			narrow, narrow, 3, index, size);

		int b;
		for(b = 0; b < noBatches ; b++){
			THTensor* selectIn = THTensor_(newSelect)(
#ifdef CUDA
				state,
#endif
				narrow, 0, b);

			THTensor* selectOut = THTensor_(newSelect)(
#ifdef CUDA
				state,
#endif
				r, 0, b);

#ifdef CUDA
			THCudaTensor_scale2d(state, selectOut, selectIn);
#else
			scale2d(selectOut, selectIn);
#endif

			THTensor_(free)(
#ifdef CUDA
				state,
#endif
				selectIn);

			THTensor_(free)(
#ifdef CUDA
				state,
#endif
				selectOut);
		}

		THTensor_(free)(
#ifdef CUDA
			state,
#endif
			narrow);

	}

	// resize to correct size
	if(noDims == 2){
		if(noBatches == 1){
			THTensor_(resize2d)(
#ifdef CUDA
				state,
#endif
				r, d[0], d[1]);
		} else {
			THTensor_(resize3d)(
#ifdef CUDA
				state,
#endif
				r, noBatches, d[0], d[1]);
		}
	} else if(noDims == 3){
		if(noBatches == 1){
			THTensor_(resize3d)(
#ifdef CUDA
				state,
#endif
				r, d[0], d[1], d[2]);
		}
	}

	env->ReleaseIntArrayElements(dims, d, 0);

	return res == NULL ? createTensorObject(env, r) : res;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_TensorOps_clamp
  (JNIEnv * env, jclass clazz, jobject res, jobject tensor, jfloat minval, jfloat maxval){
	THTensor* r = getTensor(env, res);
	THTensor* t = getTensor(env, tensor);
	THTensor_(clamp)(
#ifdef CUDA
			state,
#endif
			r, t, minval, maxval);
	return res == NULL ? createTensorObject(env, r) : res;
}

