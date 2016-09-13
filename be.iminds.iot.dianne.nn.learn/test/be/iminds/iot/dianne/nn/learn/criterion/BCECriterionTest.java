package be.iminds.iot.dianne.nn.learn.criterion;

import org.junit.Before;
import be.iminds.iot.dianne.api.nn.learn.Criterion;

public class BCECriterionTest extends AbstractCriterionTest {
	
	@Before
	public void setup() {
		out = new float[][]{{0.5f}, {0.8f}, {0.3f}, {0.3f}};
		tar = new float[][]{{0.5f}, {0.3f}, {0.8f}, {0.4f}};
		loss = new float[]{0.693147f, 1.19355f, 1.03451f, 0.695594f};
		grad = new float[][]{{0.0f}, {3.125f}, {-2.38095f}, {-0.47619f}};
	}
	
	@Override
	protected Criterion newCriterion(int outSize) {
		return new BCECriterion();
	}
	
}
