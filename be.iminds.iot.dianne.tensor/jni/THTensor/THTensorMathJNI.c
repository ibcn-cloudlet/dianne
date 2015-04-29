#include "be_iminds_iot_dianne_tensor_impl_th_THTensorMath.h"

#include "THTensorJNI.h"


JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong add) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_add__JJFJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val , jlong add) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJF(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong sub) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_sub__JJFJ(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val, jlong sub) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mul(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_cmul(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong mul) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_div(
		JNIEnv * env, jobject o, jlong dst, jlong src, jfloat val) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_cdiv(
		JNIEnv * env, jobject o, jlong dst, jlong src, jlong div) {
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_dot(
		JNIEnv * env, jobject o, jlong v1, jlong v2) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_vv(
		JNIEnv * env, jobject o, jlong dst, jlong v1, jlong v2) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_tmv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mm(
		JNIEnv * env, jobject o, jlong dst, jlong m1, jlong m2) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addvv(
		JNIEnv * env, jobject o, jlong dst, jlong m, jlong v1, jlong v2) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addmv(
		JNIEnv * env, jobject o, jlong dst, jlong v1, jlong m, jlong v2) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_addmm(
		JNIEnv * env, jobject o, jlong dst, jlong m1, jlong m2, jlong m3) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_exp(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_log(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
}

JNIEXPORT jlong JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_tanh(
		JNIEnv * env, jobject o, jlong dst, jlong src) {
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
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_max(
		JNIEnv * env, jobject o, jlong src) {
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_min(
		JNIEnv * env, jobject o, jlong src) {
}

JNIEXPORT jfloat JNICALL Java_be_iminds_iot_dianne_tensor_impl_th_THTensorMath_mean(
		JNIEnv * env, jobject o, jlong src) {
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
