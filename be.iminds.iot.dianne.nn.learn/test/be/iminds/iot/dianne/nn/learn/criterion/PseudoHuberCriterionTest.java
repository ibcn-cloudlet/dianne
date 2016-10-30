package be.iminds.iot.dianne.nn.learn.criterion;

import org.junit.Before;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.BatchConfig;

public class PseudoHuberCriterionTest extends AbstractCriterionTest {

	@Before
	public void setup() {
		out = new float[][]{{1.0f}, {0.0f}, {0.9f}, {1.1f}};
		tar = new float[][]{{1.0f}, {1.0f}, {1.0f}, {1.0f}};
		loss = new float[]{0.0f, 0.4142135623731f, 0.004987562112089f, 0.004987562112089f};
		grad = new float[][]{{0.0f}, {-0.7071067812f}, {-0.09950371902f}, {0.09950371902f}};
	}
	
	@Override
	protected Criterion newCriterion() {
		BatchConfig b = new BatchConfig();
		b.batchSize = 1;
		return new PseudoHuberCriterion(b);
	}
	
}
