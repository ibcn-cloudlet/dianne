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
package be.iminds.iot.dianne.nn.test.benchmark;

public class OverfeatFast extends DianneBenchmark {

	public void testSingleForward() throws Exception {
		long time = benchmark("overfeat_fast", new int[]{3, 231, 231}, 10, false);
		System.out.println("Overfeat Fast single forward "+time+" ms");
	}
	
	public void testBatchedForward() throws Exception {
		long time = benchmark("overfeat_fast", new int[]{10, 3, 231, 231}, 10, false);
		System.out.println("Overfeat Fast batched (10) forward "+time+" ms");
	}
	
	public void testForwardBackward() throws Exception {
		long time = benchmark("overfeat_fast", new int[]{128, 3, 231, 231}, 10, true);
		System.out.println("Overfeat Fast batched (128) forward+backward "+time+" ms");
	}
}
