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
 *     Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn.criterion;

import org.junit.Before;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.BatchConfig;

public class GaussianKLDivCriterionTest extends AbstractCriterionTest {
	
	@Before
	public void setup() {
		out = new float[][]{{0.0f, 1.0f}, {3.14f, 2.7f}, {3.14f, 2.7f}, {-3.14f, 2.7f}, {3.14f, -2.7f}};
		tar = new float[][]{{0.0f, 1.0f}, {3.14f, 2.7f}, {0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}};
		loss = new float[]{0.0f, 0.0f, 7.08155f, 7.08155f, 7.08155f};
		grad = new float[][]{{0.0f, 0.0f}, {0.0f, 0.0f}, {3.14f, 2.329629f}, {-3.14f, 2.329629f}, {3.14f, -2.329629f}};
	}
	
	@Override
	protected Criterion newCriterion() {
		BatchConfig b = new BatchConfig();
		b.batchSize = 1;
		return new GaussianKLDivCriterion(b);
	}
	
}
