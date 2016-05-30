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
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.nn.module.layer;

import org.junit.Test;

import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.tensor.Tensor;

public class SpatialAvgpoolingTest extends ModuleTest{

	@Test
	public void testSpatialAvgPooling() throws Exception {
		
		SpatialAvgPooling pool = new SpatialAvgPooling(2, 2, 2, 2); 
		
		float[] inputData = new float[]{
				-1.0016336f, -0.3659996f, -0.561763f, 0.025022956f, 0.39551026f, 0.25571227f, 
				-0.36720142f, -1.1175866f, 1.6601679f, 0.7697076f, 0.94416225f, -0.33390528f, 
				-0.7185745f, -0.82316816f, -0.4198505f, 0.8720639f, -0.91903704f, 1.1918385f, 
				0.9839494f, -1.1864451f, -0.2240562f, 0.1604332f, -1.2869335f, -0.3002753f, 
				0.70621985f, 1.6712323f, 0.427771f, -1.1304947f, 1.5576284f, 0.8286627f, 
				1.1835989f, -1.261588f, 0.109402984f, 0.31725317f, 1.2378355f, 0.40925455f, 

				-0.56402963f, -1.0522915f, -0.9422165f, 1.0155184f, -0.31722248f, 1.1515416f, 
				-0.14420497f, -1.6517973f, 0.4656332f, 1.050356f, -0.5541778f, -0.21497928f, 
				-0.2700798f, 0.7216463f, 0.10998042f, -0.21850033f, 0.58419466f, 0.59096277f, 
				-1.3353262f, 1.5876176f, -0.9639381f, 0.21641004f, 2.2216365f, -0.19380932f, 
				-0.027697943f, 0.82068586f, 0.7767722f, 0.69155896f, -1.9302452f, 0.16983992f, 
				1.1714138f, -1.4150684f, -0.7196508f, 0.74438053f, 0.88626957f, -1.6724747f
		};
		Tensor input = new Tensor(inputData, 2,6,6);
		
		float[] gradOutputData = new float[]{
				-0.024800602f, 0.022750217f, 0.22541349f, 
				-0.055356164f, -0.07583931f, 0.10012579f, 
				-0.0026466453f, 0.22068158f, 0.04035959f, 

				0.14729609f, -0.039030015f, -0.017672129f, 
				-0.007538767f, 0.22331862f, -0.076386474f, 
				0.09946432f, -0.021048022f, 0.001844853f
		};
		Tensor gradOutput = new Tensor(gradOutputData, 2,3,3);
		
		float[] expOutputData = new float[]{
				-0.7131053f, 0.4732839f, 0.3153699f,
				-0.4360596f, 0.0971476f, -0.32860184f,
				0.5748657f, -0.06901689f, 1.0083452f,
				
				-0.85308087f, 0.39732277f, 0.016290504f,
				0.17596449f, -0.214012f, 0.80074614f,
				0.13733333f, 0.3732652f, -0.6366526f
		};
		Tensor expOutput = new Tensor(expOutputData, 2,3,3);
		
		float[] expGradInputData = new float[]{
				-0.0062001506f, -0.0062001506f, 0.0056875544f, 0.0056875544f, 0.05635337f, 0.05635337f,
				-0.0062001506f, -0.0062001506f, 0.0056875544f, 0.0056875544f, 0.05635337f, 0.05635337f,
				-0.013839041f, -0.013839041f, -0.018959828f, -0.018959828f, 0.025031447f, 0.025031447f,
				-0.013839041f, -0.013839041f, -0.018959828f, -0.018959828f, 0.025031447f, 0.025031447f,
				-6.616613E-4f, -6.616613E-4f, 0.055170394f, 0.055170394f, 0.010089898f, 0.010089898f,
				-6.616613E-4f, -6.616613E-4f, 0.055170394f, 0.055170394f, 0.010089898f, 0.010089898f,
				
				0.03682402f, 0.03682402f, -0.009757504f, -0.009757504f, -0.0044180322f, -0.0044180322f,
				0.03682402f, 0.03682402f, -0.009757504f, -0.009757504f, -0.0044180322f, -0.0044180322f,
				-0.0018846918f, -0.0018846918f, 0.055829655f, 0.055829655f, -0.019096619f, -0.019096619f,
				-0.0018846918f, -0.0018846918f, 0.055829655f, 0.055829655f, -0.019096619f, -0.019096619f,
				0.02486608f, 0.02486608f, -0.0052620056f, -0.0052620056f, 4.6121326E-4f, 4.6121326E-4f,
				0.02486608f, 0.02486608f, -0.0052620056f, -0.0052620056f, 4.6121326E-4f, 4.6121326E-4f
		};
		Tensor expGradInput = new Tensor(expGradInputData, 2,6,6);
		
		testModule(pool, input, expOutput, gradOutput, expGradInput);
	}
}
