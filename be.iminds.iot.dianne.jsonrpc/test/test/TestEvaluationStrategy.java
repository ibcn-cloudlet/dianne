package test;

import java.util.Map;

import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.nn.eval.strategy.AbstractEvaluationStrategy;
import be.iminds.iot.dianne.tensor.Tensor;

public class TestEvaluationStrategy extends AbstractEvaluationStrategy {
	@Override
	protected void init(Map<String, String> config) {
		System.out.println("INIT");
	}

	protected float eval(Tensor output, Tensor target) {
		System.out.println("OUTPUTS:");
		System.out.println(output);
		System.out.println("TARGET:");
		System.out.println(target);
		return 0.0f;
	}

	@Override
	protected Evaluation finish() {
		return new Evaluation();
	}
}
