#include "be_iminds_iot_dianne_tensor_impl_th_THTensorMath.h"

#ifdef CUDA
#include "THCudaTensorJNI.h"
#include "THCudaTensorOps.h"
#else
#include "THTensorJNI.h"
#endif

THTensor* getTHTensor(THTensor* l){
	return l==0 ? THTensor_(new)(
#ifdef CUDA
			state
#endif
	) : l;
}

THTensor* getTHTensor1(THTensor* l, long d1){
	return l==0 ? THTensor_(newWithSize1d)(
#ifdef CUDA
			state,
#endif
			d1) : l;
}

THTensor* getTHTensor2(THTensor* l, long d1, long d2){
	return l==0 ? THTensor_(newWithSize2d)(
#ifdef CUDA
			state,
#endif
			d1, d2) : l;
}

THTensor* getTHTensor3(THTensor* l, long d1, long d2, long d3){
	return l==0 ? THTensor_(newWithSize3d)(
#ifdef CUDA
			state,
#endif
			d1, d2, d3) : l;
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
		THTensor* v = THTensor_(newWithTensor)(
#ifdef CUDA
				state,
#endif
				l);
		THTensor_(resize1d)(
#ifdef CUDA
				state,
#endif
				v , getSize(l));
		return v;
	}
}

void releaseVector(THTensor* l, THTensor* v){
	if(v!=l){
		THTensor_(free)(
#ifdef CUDA
				state,
#endif
				v);
	}
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(add)(
#ifdef CUDA
			state,
#endif
			r, t, val);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong add) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* a = (THTensor*) add;
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			r, t, 1, a);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJFJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val , jlong add) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* a = (THTensor*) add;
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			r, t, val, a);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(add)(
#ifdef CUDA
			state,
#endif
			r, t, -val);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong sub) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* s = (THTensor*) sub;
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			r, t, -1, s);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJFJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val, jlong sub) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* s = (THTensor*) sub;
	THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			r, t, -val, s);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mul(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(mul)(
#ifdef CUDA
			state,
#endif
			r, t, val);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_cmul(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong mul) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* m = (THTensor*) mul;
	THTensor_(cmul)(
#ifdef CUDA
			state,
#endif
			r, t, m);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_div(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(div)(
#ifdef CUDA
			state,
#endif
			r, t, val);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_cdiv(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong div) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor* d = (THTensor*) div;
	THTensor_(cdiv)(
#ifdef CUDA
			state,
#endif
			r, t, d);
	return r;
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dot(
		JNIEnv * env, jobject o, jlong v1, jlong v2) {
	THTensor* vec1 = (THTensor*) v1;
	THTensor* vec2 = (THTensor*) v2;
	return THTensor_(dot)(
#ifdef CUDA
			state,
#endif
			vec1, vec2);
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_vv(
		JNIEnv * env, jobject o, jlong dst, jlong v1, jlong v2) {
	THTensor* vec1= getVector(v1);
	THTensor* vec2 = getVector(v2);
	THTensor* r = getTHTensor2(dst, vec1->size[0], vec2->size[0]);

	THTensor_(addr)(
#ifdef CUDA
			state,
#endif
			r, 0.0f, r, 1.0f, vec1, vec2);

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


	THTensor_(addmv)(
#ifdef CUDA
			state,
#endif
			rv, 0.0f, rv, 1.0f, mat, vec);

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

	releaseVector(v, vec);
	//releaseVector(rv, r);
	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			transpose);

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mm(
		JNIEnv * env, jobject o, jlong dst, jlong m1, jlong m2) {
	THTensor* mat1 = (THTensor*) m1;
	THTensor* mat2 = (THTensor*) m2;
	THTensor* r = getTHTensor2(dst, mat1->size[0], mat2->size[1]);

	THTensor_(addmm)(
#ifdef CUDA
			state,
#endif
			r, 0.0f, r, 1.0f, mat1, mat2);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addvv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v1, jlong v2) {
	THTensor* mat = (THTensor*) m;
	THTensor* vec1 = getVector(v1);
	THTensor* vec2 = getVector(v2);
	THTensor* r = getTHTensor2(dst, mat->size[0], mat->size[1]);

	THTensor_(addr)(
#ifdef CUDA
			state,
#endif
			r, 1.0f, mat, 1.0f, vec1, vec2);

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

	THTensor_(addmv)(
#ifdef CUDA
			state,
#endif
			rv, 1.0f, vec1, 1.0f, mat, vec2);

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

	THTensor_(addmm)(
#ifdef CUDA
			state,
#endif
			r, 1.0f, mat1, 1.0f, mat2, mat3);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_exp(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
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
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_log(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
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
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_tanh(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);
	THTensor_(tanh)(
#ifdef CUDA
			state,
#endif
			r, t);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dtanh(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

#ifdef CUDA
	THCudaTensor_dtanh(state, r, t);
#else
	TH_TENSOR_APPLY2(real, r, real, t, real z = *t_data; *r_data = 1.- z * z;)
#endif

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sigmoid(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

#ifdef CUDA
	THCudaTensor_sigmoid(state, r, t);
#else
	TH_TENSOR_APPLY2(real, r, real, t, *r_data = 1./(1.+ exp(- *t_data));)
#endif

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dsigmoid(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

#ifdef CUDA
	THCudaTensor_dsigmoid(state, r, t);
#else
	TH_TENSOR_APPLY2(real, r, real, t,\
	                   real z = *t_data; \
	                   *r_data = (1. - z) * z;)
#endif

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_thresh__JJFFF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat thres, jfloat coeff, jfloat of) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

#ifdef CUDA
	THCudaTensor_threshold(state, r, t, thres, coeff, of);
#else
	TH_TENSOR_APPLY2(real, r, real, t,\
	                   real z = *t_data; \
	                   *r_data = z > thres ? z : coeff * z + of;)
#endif

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_thresh__JJJJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong t, jlong c, jlong of) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dthresh__JJFF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat thres, jfloat coeff) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);

#ifdef CUDA
	THCudaTensor_dthreshold(state, r, t, thres, coeff);
#else
	TH_TENSOR_APPLY2(real, r, real, t,\
	                   real z = *t_data; \
	                   *r_data = z > thres ? 1 : coeff;)
#endif

	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dthresh__JJJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong t, jlong c) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_softmax(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;

	THTensor_(resizeAs)(
#ifdef CUDA
			state,
#endif
			r, t);
	float max = THTensor_(maxall)(
#ifdef CUDA
			state,
#endif
			t);
#ifdef CUDA
	THCudaTensor_expminus(state, r, t, max);
#else
	TH_TENSOR_APPLY2(real, r, real, t,\
	                   *r_data = exp(*t_data - max);)
#endif

	float sum = THTensor_(sumall)(
#ifdef CUDA
			state,
#endif
			r);
	THTensor_(div)(
#ifdef CUDA
			state,
#endif
			r, r, sum);

	return r;
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sum(
		JNIEnv * env, jobject o, jlong src) {
	THTensor* t = (THTensor*) src;
	return THTensor_(sumall)(
#ifdef CUDA
			state,
#endif
			t);
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_max(
		JNIEnv * env, jobject o, jlong src) {
	THTensor* t = (THTensor*) src;
	return THTensor_(maxall)(
#ifdef CUDA
			state,
#endif
			t);
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_min(
		JNIEnv * env, jobject o, jlong src) {
	THTensor* t = (THTensor*) src;
	return THTensor_(minall)(
#ifdef CUDA
			state,
#endif
			t);
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mean(
		JNIEnv * env, jobject o, jlong src) {
	THTensor* t = (THTensor*) src;
	return THTensor_(meanall)(
#ifdef CUDA
			state,
#endif
			t);
}

JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_argmax(
		JNIEnv * env, jobject o, jlong src) {
	// only works for 1 dim tensors
	THTensor* t = (THTensor*) src;

	int index = 0;

#ifdef CUDA
	index = THCudaTensor_argmax(state, t);
#else
	real* data_ptr = THTensor_(data)(t);
	real max = data_ptr[0];
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

#endif
	return index;
}

JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_argmin(
		JNIEnv * env, jobject o, jlong src) {
	// only works for 1 dim tensors
	THTensor* t = (THTensor*) src;

	int index = 0;

#ifdef CUDA
	index = THCudaTensor_argmin(state, t);
#else
	real* data_ptr = THTensor_(data)(t);
	real min = data_ptr[0];
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

#endif
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

	long outputWidth;
	long outputHeight;
	char type[2];

	if(mode==1){
		// full
		outputWidth  = inputWidth + (kernelWidth - 1);
		outputHeight = inputHeight + (kernelHeight - 1);

#ifdef CUDA
		type[0] = 'f';
		type[1] = 'x';
#else
		type[0] = 'F';
#endif

	} else {
		// valid
		outputWidth  = (inputWidth - kernelWidth) / sx + 1;
		outputHeight = (inputHeight - kernelHeight) / sy + 1;
#ifdef CUDA
		type[0] = 'v';
		type[1] = 'x';
#else
		type[0] = 'V';
#endif
	}

	// initialize output with add or zero
	THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			r, outputHeight, outputWidth);
	if(add!=0){
		THTensor_(copy)(
#ifdef CUDA
				state,
#endif
				r, add);
	} else {
	    THTensor_(zero)(
#ifdef CUDA
	    		state,
#endif
	    		r);
	}

	// resize to 3d in/out and use conv2Dmv
	THTensor_(resize3d)(
#ifdef CUDA
			state,
#endif
			r, 1, outputHeight, outputWidth);
	THTensor_(resize3d)(
#ifdef CUDA
			state,
#endif
			t, 1, inputHeight, inputWidth);
	THTensor_(resize4d)(
#ifdef CUDA
			state,
#endif
			kernel, 1, 1, kernelHeight, kernelWidth);

#ifdef CUDA
	THTensor_(conv2Dmv)(state, r, 1.0, t, kernel, sy, sx, type);
#else
	THTensor_(conv2Dmv)(r, 1.0, 1.0, t, kernel, sy, sx, type, "X");
#endif


	// resize back to 2d tensors
	THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			r, outputHeight, outputWidth);
	THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			t, inputHeight, inputWidth);
	THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			kernel, kernelHeight, kernelWidth);

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

#ifdef CUDA
	// TODO implement CUDA
#else

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
#endif
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
	THTensor_(resize3d)(
#ifdef CUDA
			state,
#endif
			output, nOutputPlane, outputHeight, outputWidth);

	/* set output to bias */
	long i;
	THTensor* temp = THTensor_(newWithSize2d)(
#ifdef CUDA
			state,
#endif
			outputHeight, outputWidth);
	for (i=0; i<bias->size[0]; i++){
		THTensor_(select)(
#ifdef CUDA
				state,
#endif
				temp, output, 0, i);
		float b = THTensor_(get1d)(
#ifdef CUDA
				state,
#endif
				bias, i);
		THTensor_(fill)(
#ifdef CUDA
				state,
#endif
				temp, b);
	}
	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			temp);

	/* do convolutions */
#ifdef CUDA
	char type[] = "vx";
	THTensor_(conv2Dmv)(state, output, 1.0, input, weight, sy, sx, type);
#else
	THTensor_(conv2Dmv)(output, 1.0, 1.0, input, weight, sy, sx, "V", "X");
#endif
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

	THTensor_(resizend)(
#ifdef CUDA
			state,
#endif
			r, noDims, newDims);
	THTensor_(zero)(
#ifdef CUDA
			state,
#endif
			r);

	THTensor* narrowed = THTensor_(newWithTensor)(
#ifdef CUDA
			state,
#endif
			r);
	// now narrow and copy
	for(i=0;i<noDims;i++){
		THTensor_(narrow)(
#ifdef CUDA
				state,
#endif
				narrowed, narrowed, i, p[i], t->size[i]);
	}
	THTensor_(copy)(
#ifdef CUDA
			state,
#endif
			narrowed, t);
	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			narrowed);

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

#ifdef CUDA
	// TODO implement CUDA
#else

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
#endif
	return r;
}
