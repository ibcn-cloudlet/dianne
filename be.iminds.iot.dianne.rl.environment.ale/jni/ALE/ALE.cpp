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


#include "be_iminds_iot_dianne_rl_environment_ale_ArcadeLearningEnvironment.h"
#include <ale_interface.hpp>
#include <iostream>

using namespace std;

ALEInterface* ALE;

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_rl_environment_ale_ArcadeLearningEnvironment_loadROM
  (JNIEnv * env, jobject o, jstring rom){
	 if(ALE == NULL){
		 ALE = new ALEInterface();
	 }

	 const char *romString = env->GetStringUTFChars(rom, 0);
	 ALE->loadROM(romString);
	 env->ReleaseStringUTFChars(rom, romString);

	 return;
}


JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_rl_environment_ale_ArcadeLearningEnvironment_getActions
  (JNIEnv * env, jobject o){
	 if(ALE == NULL){
		 ALE = new ALEInterface();
	 }
	 ActionVect minimal_actions = ALE->getMinimalActionSet();
	 return minimal_actions.size();
}


JNIEXPORT jint JNICALL Java_be_iminds_iot_dianne_rl_environment_ale_ArcadeLearningEnvironment_performAction
  (JNIEnv * env, jobject o, jint action){
	return ALE->act(ALE->getMinimalActionSet()[action]);
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_rl_environment_ale_ArcadeLearningEnvironment_resetGame
  (JNIEnv * env, jobject o){
	ALE->reset_game();
}


JNIEXPORT jboolean JNICALL Java_be_iminds_iot_dianne_rl_environment_ale_ArcadeLearningEnvironment_gameOver
  (JNIEnv * env, jobject o){
	return ALE->game_over();
}


JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_rl_environment_ale_ArcadeLearningEnvironment_setFrameskip
  (JNIEnv * env, jobject o, jint skip){
	if(ALE == NULL){
		ALE = new ALEInterface();
	}

	ALE->setInt("frame_skip", skip);
}


JNIEXPORT jfloatArray JNICALL Java_be_iminds_iot_dianne_rl_environment_ale_ArcadeLearningEnvironment_getScreen
  (JNIEnv * env, jobject o){
	ALEScreen screen = ALE->getScreen();

	pixel_t* screen_data = screen.getArray();
	pixel_t* s = screen_data;

	int pixels = 33600;
	int size = 3*pixels;
	jfloatArray result;
	result = env->NewFloatArray(size);

	int k, i;
	jfloat data[size];
	jfloat* ptr_r = data;
	jfloat* ptr_g = &data[pixels];
	jfloat* ptr_b = &data[2*pixels];

	int r,g,b;
	for (i = 0; i < pixels; i++) {
		pixel_t pixel = *s++;

		ALE->theOSystem->colourPalette().getRGB(pixel, r, g, b);

		*ptr_r++ = r / 255.0f;
		*ptr_g++ = g / 255.0f;
		*ptr_b++ = b / 255.0f;
	}

	env->SetFloatArrayRegion(result, 0, size, data);

	return result;
}
