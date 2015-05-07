#include "be_iminds_iot_dianne_tensor_impl_th_THTensorMath.h"

#include "THTensorJNI.h"

THTensor* getTHTensor(THTensor* l){
	return l==0 ? THTensor_(new)() : l;
}

THTensor* getTHTensor1(THTensor* l, long d1){
	return l==0 ? THTensor_(newWithSize1d)(d1) : l;
}

THTensor* getTHTensor2(THTensor* l, long d1, long d2){
	return l==0 ? THTensor_(newWithSize2d)(d1, d2) : l;
}

THTensor* getTHTensor3(THTensor* l, long d1, long d2, long d3){
	return l==0 ? THTensor_(newWithSize3d)(d1, d2, d3) : l;
}

long int getSize(THTensor* t){
	long int size = 1;
	long* ptr = t->size;
	int i=0;
	for(i=0;i<t->nDimension;i++){
		size =size*(*(ptr++));
	}
	return size;
}

THTensor* getVector(THTensor* l){
	if(l->nDimension==1){
		return l;
	} else {
		THTensor* v = THTensor_(newWithTensor)(l);
		THTensor_(resize1d)(v , getSize(l));
		return v;
	}
}

void releaseVector(THTensor* l, THTensor* v){
	if(v!=l){
		THTensor_(free)(v);
	}
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(add)(r, t, val);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong add) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* a = (THTensor*) add;
	THTensor_(cadd)(r, t, 1, a);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJFJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val , jlong add) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* a = (THTensor*) add;
	THTensor_(cadd)(r, t, val, a);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(add)(r, t, -val);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong sub) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* s = (THTensor*) sub;
	THTensor_(cadd)(r, t, -1, s);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJFJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val, jlong sub) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* s = (THTensor*) sub;
	THTensor_(cadd)(r, t, -val, s);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mul(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(mul)(r, t, val);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_cmul(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong mul) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* m = (THTensor*) mul;
	THTensor_(cmul)(r, t, m);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_div(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(div)(r, t, val);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_cdiv(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong div) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* d = (THTensor*) div;
	THTensor_(cdiv)(r, t, d);
	return r;
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dot(
		JNIEnv * env, jobject o, jlong v1, jlong v2) {
	THTensor* vec1 = (THTensor*) v1;
	THTensor* vec2 = (THTensor*) v2;
	return THTensor_(dot)(vec1, vec2);
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_vv(
		JNIEnv * env, jobject o, jlong dst, jlong v1, jlong v2) {
	THTensor* vec1= getVector(v1);
	THTensor* vec2 = getVector(v2);
	THTensor* r = getTHTensor2(dst, vec1->size[0], vec2->size[0]);

	THTensor_(addr)(r, 0.0f, r, 1.0f, vec1, vec2);

	releaseVector(v1, vec1);
	releaseVector(v2, vec2);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v) {
	THTensor* mat = (THTensor*) m;
	THTensor* vec = getVector(v);

	THTensor* r = getTHTensor1(dst, mat->size[0]);
	THTensor* rv = getVector(r);


	THTensor_(addmv)(rv, 0.0f, rv, 1.0f, mat, vec);

	releaseVector(v, vec);
	//releaseVector(rv, r);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_tmv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v) {
	THTensor* mat = (THTensor*) m;
	THTensor* vec = getVector(v);
	THTensor* r = getTHTensor1(dst, mat->size[1]);
	THTensor* rv = getVector(r);

	THTensor* transpose = THTensor_(newTranspose)(mat, 0, 1);

	THTensor_(addmv)(rv, 0.0f, rv, 1.0f, transpose, vec);

	releaseVector(v, vec);
	//releaseVector(rv, r);
	THTensor_(free)(transpose);

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mm(
		JNIEnv * env, jobject o, jlong dst, jlong m1, jlong m2) {
	THTensor* mat1 = (THTensor*) m1;
	THTensor* mat2 = (THTensor*) m2;
	THTensor* r = getTHTensor2(dst, mat1->size[0], mat2->size[1]);

	THTensor_(addmm)(r, 0.0f, r, 1.0f, mat1, mat2);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addvv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v1, jlong v2) {
	THTensor* mat = (THTensor*) m;
	THTensor* vec1 = getVector(v1);
	THTensor* vec2 = getVector(v2);
	THTensor* r = getTHTensor2(dst, mat->size[0], mat->size[1]);

	THTensor_(addr)(r, 1.0f, mat, 1.0f, vec1, vec2);

	releaseVector(v1, vec1);
	releaseVector(v2, vec2);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addmv(
		JNIEnv * env, jobject o, jlong dst, jlong v1, jlong m, jlong v2) {
	THTensor* mat = (THTensor*) m;
	THTensor* vec1 = getVector(v1);
	THTensor* vec2 = getVector(v2);
	THTensor* r = getTHTensor1(dst, vec1->size[0]);
	THTensor* rv = getVector(r);

	THTensor_(addmv)(rv, 1.0f, vec1, 1.0f, mat, vec2);

	releaseVector(v1, vec1);
	releaseVector(v2, vec2);
	//releaseVector(rv, r);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addmm(
		JNIEnv * env, jobject o, jlong dst, jlong m1, jlong m2, jlong m3) {
	THTensor* mat1 = (THTensor*) m1;
	THTensor* mat2 = (THTensor*) m2;
	THTensor* mat3 = (THTensor*) m3;
	THTensor* r = getTHTensor2(dst, mat3->size[0], mat3->size[1]);

	THTensor_(addmm)(r, 1.0f, mat1, 1.0f, mat2, mat3);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_exp(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);
	THTensor_(exp)(r, t);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_log(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);
	THTensor_(log)(r, t);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_tanh(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);
	THTensor_(tanh)(r, t);
	return r;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dtanh(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);

	TH_TENSOR_APPLY2(real, r, real, t, real z = *t_data; *r_data = 1.- z * z;)

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sigmoid(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);

	TH_TENSOR_APPLY2(real, r, real, t, *r_data = 1./(1.+ exp(- *t_data));)

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dsigmoid(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);

	TH_TENSOR_APPLY2(real, r, real, t,\
	                   real z = *t_data; \
	                   *r_data = (1. - z) * z;)

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_thresh__JJFFF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat thres, jfloat coeff, jfloat of) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);

	TH_TENSOR_APPLY2(real, r, real, t,\
	                   real z = *t_data; \
	                   *r_data = z > thres ? z : coeff * z + of;)

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_thresh__JJJJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong t, jlong c, jlong of) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dthresh__JJFF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat thres, jfloat coeff) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);

	TH_TENSOR_APPLY2(real, r, real, t,\
	                   real z = *t_data; \
	                   *r_data = z > thres ? 1 : coeff;)

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dthresh__JJJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong t, jlong c) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_softmax(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(r, t);
	float max = THTensor_(maxall)(t);

	TH_TENSOR_APPLY2(real, r, real, t,\
	                   *r_data = exp(*t_data - max);)

	float sum = THTensor_(sumall)(r);
	THTensor_(div)(r, r, sum);

	return r;
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sum(
		JNIEnv * env, jobject o, jlong src) {
	THTensor* t = (THTensor*) src;
	return THTensor_(sumall)(t);
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_max(
		JNIEnv * env, jobject o, jlong src) {
	THTensor* t = (THTensor*) src;
	return THTensor_(maxall)(t);
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_min(
		JNIEnv * env, jobject o, jlong src) {
	THTensor* t = (THTensor*) src;
	return THTensor_(minall)(t);
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mean(
		JNIEnv * env, jobject o, jlong src) {
	THTensor* t = (THTensor*) src;
	return THTensor_(meanall)(t);
}

JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_argmax(
		JNIEnv * env, jobject o, jlong src) {
	// only works for 1 dim tensors
	THTensor* t = (THTensor*) src;
	real* data_ptr = THTensor_(data)(t);

	real max = data_ptr[0];
	int index = 0;
	long size = t->size[0];
	long stride = t->stride[0];
	int i = 0;
	for(i=0;i<size;i++){
		if(max < *data_ptr){
			max = *data_ptr;
			index = i;
		}
		data_ptr += stride;
	}
	return index;
}

JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_argmin(
		JNIEnv * env, jobject o, jlong src) {
	// only works for 1 dim tensors
	THTensor* t = (THTensor*) src;
	real* data_ptr = THTensor_(data)(t);

	real min = data_ptr[0];
	int index = 0;
	long size = t->size[0];
	long stride = t->stride[0];
	int i = 0;
	for(i=0;i<size;i++){
		if(min > *data_ptr){
			min = *data_ptr;
			index = i;
		}
		data_ptr += stride;
	}
	return index;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_convolution2D(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong k, jint sx, jint sy, jint mode,
		jboolean flip) {
	Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addconvolution2D(
			env, o, dst, 0, src, k, sx, sy, mode, flip);
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addconvolution2D(
		JNIEnv * env, jobject o, jlong dst, jlong add, jlong src, jlong k, jint sx, jint sy, jint mode,
		jboolean flip) {
	THTensor* r = getTHTensor(dst);

	THTensor* t = (THTensor*) src;
	THTensor* kernel = (THTensor*) k;

	long kernelWidth = kernel->size[1];
	long kernelHeight = kernel->size[0];
	long inputWidth = t->size[1];
	long inputHeight = t->size[0];

	if(flip){
		// TODO flip kernel?
	}

	if(mode==1){
		// full
		long outputWidth  = inputWidth + (kernelWidth - 1);
		long outputHeight = inputHeight + (kernelHeight - 1);
		THTensor_(resize2d)(r, outputHeight, outputWidth);
	    if(add!=0){
	    	THTensor_(copy)(r, add);
	    } else {
	    	THTensor_(zero)(r);
	    }

		THTensor_(fullConv2Dptr)(THTensor_(data)(r),
				1.0f, THTensor_(data)(t), inputHeight, inputWidth,
				THTensor_(data)(kernel), kernelHeight, kernelWidth,
				sy, sx);
	} else if(mode==2){
		// same
		// add zero padding first
		int pad_h = (kernelHeight - 1)/2;
		int pad_w = (kernelWidth - 1)/2;

		int paddedWidth = inputWidth + 2*pad_w;
		int paddedHeight = inputHeight + 2*pad_h;

	    THTensor* padded = THTensor_(newWithSize2d)(paddedHeight,paddedWidth);
	    THTensor_(zero)(padded);

	    THTensor* pad_narrow = THTensor_(newWithTensor)(padded);
	    THTensor_(narrow)(pad_narrow, padded, 0, pad_h, inputHeight);
	    THTensor_(narrow)(pad_narrow, pad_narrow, 1, pad_w, inputWidth);
	    THTensor_(copy)(pad_narrow, t);

	    THTensor_(resize2d)(r, inputHeight, inputWidth);
	    if(add!=0){
	    	THTensor_(copy)(r, add);
	    } else {
	    	THTensor_(zero)(r);
	    }

	    THTensor_(validConv2Dptr)(THTensor_(data)(r),
	    				1.0f, THTensor_(data)(padded), paddedHeight, paddedWidth,
	    				THTensor_(data)(kernel), kernelHeight, kernelWidth,
	    				sy, sx);

	    THTensor_(free)(pad_narrow);
	    THTensor_(free)(padded);

	} else {
		// valid
		long outputWidth  = (inputWidth - kernelWidth) / sx + 1;
		long outputHeight = (inputHeight - kernelHeight) / sy + 1;
		THTensor_(resize2d)(r, outputHeight, outputWidth);
	    if(add!=0){
	    	THTensor_(copy)(r, add);
	    } else {
	    	THTensor_(zero)(r);
	    }
		THTensor_(validConv2Dptr)(THTensor_(data)(r),
				1.0f, THTensor_(data)(t), inputHeight, inputWidth,
				THTensor_(data)(kernel), kernelHeight, kernelWidth,
				sy, sx);
	}

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_maxpool2D(
		JNIEnv * env, jobject o, jlong dst, jlong src, jint w, jint h, jint sx, jint sy) {

	THTensor* t = (THTensor*) src;
	int iwidth = t->size[1];
	int iheight = t->size[0];

	int owidth = (iwidth - w)/sx + 1;
	int oheight = (iheight - h)/sy + 1;

	THTensor* r = getTHTensor2(dst, oheight, owidth);

	// impl from torch
	real* input_p = THTensor_(data)(t);
	real* output_p = THTensor_(data)(r);

	long i, j;
	for (i = 0; i < oheight; i++) {
		for (j = 0; j < owidth; j++) {
			real *ip = input_p + i * iwidth * sy + j * sx;
			real *op = output_p + i * owidth + j;

			real maxval = -THInf;
			long tcntr = 0;
			int x, y;
			for (y = 0; y < h; y++) {
				for (x = 0; x < w; x++) {
					real val = *(ip + y * iwidth + x);
					if (val > maxval) {
						maxval = val;
					}
					tcntr++;
				}
			}

			*op = maxval;
		}
	}

	return r;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dmaxpool2D(
		JNIEnv * env, jobject o, jlong dst, jlong m2, jlong m1, jint w, jint h, jint sx, jint sy) {
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_spatialconvolve
  (JNIEnv * env, jobject o, jlong dst, jlong add, jlong src, jlong k, jint sx, jint sy){
	// based on torch/overfeat impl
	THTensor* output = getTHTensor(dst);
	THTensor* input = (THTensor*) src;
	THTensor* weight = (THTensor*) k;
	THTensor* bias = (THTensor*) add;

	long nOutputPlane = weight->size[0];
	long kW = weight->size[3];
	long kH = weight->size[2];
	long inputWidth = input->size[2];
	long inputHeight = input->size[1];
	long outputWidth = (inputWidth - kW) / sx + 1;
	long outputHeight = (inputHeight - kH) / sy + 1;

	long i;
	real* bias_data;
	real* output_data;

	THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
	/* add bias */
	bias_data = THTensor_(data)(bias);
	output_data = THTensor_(data)(output);

#pragma omp parallel for private(i)
	for (i = 0; i < bias->size[0]; i++) {
		real *ptr_output = output_data + i * outputWidth * outputHeight;
		long j;
		for (j = 0; j < outputWidth * outputHeight; j++)
			ptr_output[j] = bias_data[i];
	}

	/* do convolutions */
	THTensor_(conv2Dmv)(output, 1.0, 1.0, input, weight, sy, sx, "V", "X");

	return output;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_zeropad
  (JNIEnv * env, jobject o, jlong dst, jlong src, jintArray paddings){
	THTensor* t = (THTensor*) src;
	THTensor* r = getTHTensor(dst);

	long newDims[t->nDimension];

	jsize noDims = (*env)->GetArrayLength(env, paddings);
	// TODO should equal t->nDimension
	jint *p = (*env)->GetIntArrayElements(env, paddings, 0);

	int i;
	for(i=0;i<noDims;i++){
		newDims[i] = t->size[i] + p[i]*2;
	}

	THTensor_(resizend)(r, noDims, newDims);
	THTensor_(zero)(r);

	THTensor* narrowed = THTensor_(newWithTensor)(r);
	// now narrow and copy
	for(i=0;i<noDims;i++){
		THTensor_(narrow)(narrowed, narrowed, i, p[i], t->size[i]);
	}
	THTensor_(copy)(narrowed, t);
	THTensor_(free)(narrowed);

	(*env)->ReleaseIntArrayElements(env, paddings, p, 0);

	return r;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_spatialmaxpool
  (JNIEnv * env, jobject o, jlong dst, jlong src, jint w, jint h, jint sx, jint sy){
	THTensor* t = (THTensor*) src;
	int noPlanes = t->size[0];
	int iwidth = t->size[2];
	int iheight = t->size[1];

	int owidth = (iwidth - w)/sx + 1;
	int oheight = (iheight - h)/sy + 1;

	THTensor* r = getTHTensor3(dst, noPlanes, oheight, owidth);

	int k;
#pragma omp parallel for private(k)
	for (k = 0; k < noPlanes; k++) {

		// TODO this is similar code as maxpool2d...
		real* input_p = THTensor_(data)(t);
		real* output_p = THTensor_(data)(r);

		long i, j;
		for (i = 0; i < oheight; i++) {
			for (j = 0; j < owidth; j++) {
				real *ip = input_p +  k*iwidth*iheight + i*iwidth*sy+ j*sx;
				real *op = output_p + k*owidth*oheight + i*owidth + j;

				real maxval = -THInf;
				long tcntr = 0;
				int x, y;
				for (y = 0; y < h; y++) {
					for (x = 0; x < w; x++) {
						real val = *(ip + y * iwidth + x);
						if (val > maxval) {
							maxval = val;
						}
						tcntr++;
					}
				}
				*op = maxval;
			}
		}
	}

	return r;
}
