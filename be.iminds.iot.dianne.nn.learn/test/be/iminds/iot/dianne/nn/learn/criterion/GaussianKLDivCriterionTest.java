package be.iminds.iot.dianne.nn.learn.criterion;

import org.junit.Before;
import be.iminds.iot.dianne.api.nn.learn.Criterion;

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
		return new GaussianKLDivCriterion();
	}
	
}
