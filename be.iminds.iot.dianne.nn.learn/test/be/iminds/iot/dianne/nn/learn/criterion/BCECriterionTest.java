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

public class BCECriterionTest extends AbstractCriterionTest {
	
	@Before
	public void setup() {
		out = new float[][]{{0.5f}, {0.8f}, {0.3f}, {0.3f}};
		tar = new float[][]{{0.5f}, {0.3f}, {0.8f}, {0.4f}};
		loss = new float[]{0.693147f, 1.19355f, 1.03451f, 0.695594f};
		grad = new float[][]{{0.0f}, {3.125f}, {-2.38095f}, {-0.47619f}};
	}
	
	@Override
	protected Criterion newCriterion() {
		BatchConfig b = new BatchConfig();
		b.batchSize = 1;
		return new BCECriterion(b);
	}
	
}
