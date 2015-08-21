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
	THTensor* v = THTensor_(newContiguous)(
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

void releaseVector(THTensor* v){
	THTensor_(free)(
#ifdef CUDA
			state,
#endif
			v);
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

	releaseVector(vec1);
	releaseVector(vec2);
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

	releaseVector(vec);
	releaseVector(rv);
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

	releaseVector(vec);
	releaseVector(rv);
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

	releaseVector(vec1);
	releaseVector(vec2);
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

	releaseVector(vec1);
	releaseVector(vec2);
	releaseVector(rv);
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

#ifdef CUDA
	THCudaTensor_softmax(state, r, t);
#else
	float max = THTensor_(maxall)(t);
	TH_TENSOR_APPLY2(real, r, real, t,\
	                   *r_data = exp(*t_data - max);)
	float sum = THTensor_(sumall)(r);
	THTensor_(div)(r, r, sum);
#endif

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
	int inputWidth = t->size[1];
	int inputHeight = t->size[0];

	int outputWidth = (inputWidth - w)/sx + 1;
	int outputHeight = (inputHeight - h)/sy + 1;

	// use 3d tensor here to use spatialmaxpool function
	THTensor* r = getTHTensor3(dst, 1, outputHeight, outputWidth);
	THTensor_(resize3d)(
#ifdef CUDA
			state,
#endif
			t, 1, inputHeight, inputWidth);

	spatialmaxpool(r, t, w, h, sx, sy);

	THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			t, inputHeight, inputWidth);
	THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			r, outputHeight, outputWidth);

	return r;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dmaxpool2D(
		JNIEnv * env, jobject o, jlong dst, jlong m2, jlong m1, jint w, jint h, jint sx, jint sy) {
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_spatialconvolve
  (JNIEnv * env, jobject o, jlong dst, jlong add, jlong src, jlong k, jint sx, jint sy, jint px, jint py){
	THTensor* output = getTHTensor(dst);
	THTensor* input = (THTensor*) src;
	THTensor* weight = (THTensor*) k;
	THTensor* bias = (THTensor*) add;

	long nOutputPlane = weight->size[0];
	long kW = weight->size[3];
	long kH = weight->size[2];
	long inputWidth = input->size[2];
	long inputHeight = input->size[1];
	long outputWidth = (inputWidth + 2*px - kW) / sx + 1;
	long outputHeight = (inputHeight + 2*py - kH) / sy + 1;
	THTensor_(resize3d)(
#ifdef CUDA
			state,
#endif
			output, nOutputPlane, outputHeight, outputWidth);

#ifdef CUDA
	// use separate cunn implementation
	THCudaTensor_spatialconvolve(state, output, input,
			weight, bias, sx, sy, px, py);
#else
	// if px,py!=0 : add padding to input
	if(px!=0 || py!=0){
		THTensor* padded = THTensor_(new)(
		#ifdef CUDA
					state
		#endif
		);
		int p[3] = {0, py, px};
		zeropad(padded, input, p);
		input = padded;
	}

	// based on torch/overfeat impl

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

	// if padded, free padded input
	if(px!=0 || py!=0){
		THTensor_(free)(
	#ifdef CUDA
				state,
	#endif
				input);
	}

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

	zeropad(r, t, p);

	(*env)->ReleaseIntArrayElements(env, paddings, p, 0);

	return r;
}

// convenience function to also use in spatialconvolve with padding
void zeropad(THTensor* r, THTensor* t, int* p){
	long newDims[t->nDimension];
	int i;
	for(i=0;i<t->nDimension;i++){
		newDims[i] = t->size[i] + p[i]*2;
	}
	THTensor_(resizend)(
#ifdef CUDA
			state,
#endif
			r, t->nDimension, newDims);
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
	for(i=0;i<t->nDimension;i++){
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
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_spatialmaxpool
  (JNIEnv * env, jobject o, jlong dst, jlong src, jint w, jint h, jint sx, jint sy){
	THTensor* t = (THTensor*) src;
	THTensor* r = getTHTensor3(dst, t->size[0], (t->size[1] - h)/sy+1, (t->size[2]-w)/sx + 1);


#ifdef CUDA
	THCudaTensor_spatialmaxpool(state, r, t, w, h, sx, sy);
#else
	int noPlanes = t->size[0];
	int iwidth = t->size[2];
	int iheight = t->size[1];

	int owidth = (iwidth - w)/sx + 1;
	int oheight = (iheight - h)/sy + 1;

	int k;
#pragma omp parallel for private(k)
	for (k = 0; k < noPlanes; k++) {

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

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_spatialdmaxpool
  (JNIEnv * env, jobject o, jlong dst, jlong src2, jlong src1, jint w, jint h, jint sx, jint sy){
	THTensor* t = (THTensor*) src1;
	THTensor* t2 = (THTensor*) src2;
	THTensor* r = getTHTensor3(dst, t->size[0], t->size[1], t->size[2]);

	THTensor_(zero)(
#ifdef CUDA
			state,
#endif
			r);

#ifdef CUDA
	THCudaTensor_spatialdmaxpool(state, r, t2, t1, w, h, sx, sy);
#else
	int noPlanes = t->size[0];
	int iwidth = t->size[2];
	int iheight = t->size[1];

	int owidth = (iwidth - w)/sx + 1;
	int oheight = (iheight - h)/sy + 1;

	int k;
#pragma omp parallel for private(k)
	for (k = 0; k < noPlanes; k++) {

		real* input_p = THTensor_(data)(t);
		real* output_p = THTensor_(data)(r);
		real* gradoutput_p = THTensor_(data)(t2);


		long i, j;
		for (i = 0; i < oheight; i++) {
			for (j = 0; j < owidth; j++) {
				real *ip = input_p +  k*iwidth*iheight + i*iwidth*sy+ j*sx;
				real *op = output_p +  k*iwidth*iheight + i*iwidth*sy+ j*sx;
				real *gop = gradoutput_p + k*owidth*oheight + i*owidth + j;

				real maxval = -THInf;
				int maxoffset = 0;
				long tcntr = 0;
				int x, y;
				for (y = 0; y < h; y++) {
					for (x = 0; x < w; x++) {
						real val = *(ip + y * iwidth + x);
						if (val > maxval) {
							maxval = val;
							maxoffset = y * iwidth + x;
						}
						tcntr++;
					}
				}
				*(op + maxoffset) = *(gop);
			}
		}
	}
#endif

	return r;

}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_scale2d
  (JNIEnv * env, jobject o, jlong dst, jlong src, jintArray dims){
	THTensor* t = (THTensor*) src;
	THTensor* r;

	jsize noDims = (*env)->GetArrayLength(env, dims);
	jint *d = (*env)->GetIntArrayElements(env, dims, 0);

	if(noDims==2){
		r = getTHTensor3(dst, 1, d[0], d[1]);
	} else {
		r = getTHTensor3(dst, d[0], d[1], d[2]);
	}

#ifdef CUDA
	THCudaTensor_scale2d(state, r, t);
#else

	int y_in = t->size[t->nDimension-2];
	int x_in = t->size[t->nDimension-1];

	int y_out = d[noDims-2];
	int x_out = d[noDims-1];

	float s_y = (y_in-1)/(float)(y_out-1);
	float s_x = (x_in-1)/(float)(x_out-1);

	int channels = r->size[0];

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
#endif

	if(noDims==2){
		THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			r, d[0], d[1]);
	}

	(*env)->ReleaseIntArrayElements(env, dims, d, 0);

	return r;
}
