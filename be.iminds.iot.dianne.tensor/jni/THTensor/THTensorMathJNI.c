#include "be_iminds_iot_dianne_tensor_impl_th_THTensorMath.h"

#include "THTensorJNI.h"

THTensor* getTHTensor(jlong l){
	return l==0 ? THTensor_(new)() : (THTensor*)l;
}

THTensor* getTHTensor1(jlong l, long d1){
	return l==0 ? THTensor_(newWithSize1d)(d1) : (THTensor*)l;
}

THTensor* getTHTensor2(jlong l, long d1, long d2){
	return l==0 ? THTensor_(newWithSize2d)(d1, d2) : (THTensor*)l;
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
	THTensor* vec1= (THTensor*) v1;
	THTensor* vec2 = (THTensor*) v2;
	THTensor* r = getTHTensor2(dst, vec1->size[0], vec2->size[0]);

	THTensor_(addr)(r, 0.0f, r, 1.0f, vec1, vec2);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v) {
	THTensor* mat = (THTensor*) m;
	THTensor* vec = (THTensor*) v;

	THTensor* r = getTHTensor1(dst, mat->size[0]);

	THTensor_(addmv)(r, 0.0f, r, 1.0f, mat, vec);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_tmv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v) {
	THTensor* mat = (THTensor*) m;
	THTensor* vec = (THTensor*) v;
	THTensor* r = getTHTensor1(dst, mat->size[1]);

	THTensor_(addmv)(r, 0.0f, r, 1.0f, THTensor_(newTranspose)(mat, 0, 1), vec);
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
	THTensor* vec1 = (THTensor*) v1;
	THTensor* vec2 = (THTensor*) v2;
	THTensor* r = getTHTensor2(dst, mat->size[0], mat->size[1]);

	THTensor_(addr)(r, 1.0f, mat, 1.0f, vec1, vec2);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addmv(
		JNIEnv * env, jobject o, jlong dst, jlong v1, jlong m, jlong v2) {
	THTensor* mat = (THTensor*) m;
	THTensor* vec1 = (THTensor*) v1;
	THTensor* vec2 = (THTensor*) v2;
	THTensor* r = getTHTensor1(dst, vec1->size[0]);

	THTensor_(addmv)(r, 1.0f, vec1, 1.0f, mat, vec2);
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
	THTensor_(exp)(r, t);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_log(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(log)(r, t);
	return r;
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_tanh(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
	THTensor* r = getTHTensor(dst);
	THTensor* t = (THTensor*) src;
	THTensor_(tanh)(r, t);
	return r;
}


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dtanh(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sigmoid(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dsigmoid(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_thresh__JJFFF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat t, jfloat c, jfloat of) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_thresh__JJJJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong t, jlong c, jlong of) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dthresh__JJFF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat t, jfloat c) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dthresh__JJJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong t, jlong c) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_softmax(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
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
}

JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_argmin(
		JNIEnv * env, jobject o, jlong src) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_convolution2D(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong k, jint sx, jint sy, jint mode,
		jboolean flip) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addconvolution2D(
		JNIEnv * env, jobject o, jlong dst, jlong add, jlong src, jlong k, jint sx, jint sy, jint mode,
		jboolean flip) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_maxpool2D(
		JNIEnv * env, jobject o, jlong dst, jlong src, jint w, jint h, jint sx, jint sy) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dmaxpool2D(
		JNIEnv * env, jobject o, jlong dst, jlong m2, jlong m1, jint w, jint h, jint sx, jint sy) {
}
