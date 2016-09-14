package be.iminds.iot.dianne.dataset.adapters;

import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This Dataset adapter modifies the Dataset by binarizing the input (image)
 * 
 * Configure by providing a threshold.
 * 
 * @author smbohez
 *
 */
@Component(
		service={Dataset.class},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.adapters.BinarizeAdapter")
public class DatasetBinarizeAdapter extends AbstractDatasetAdapter {
	
	// Threshold
	private float threshold = 0.5f;
	
	// Placeholder
	private Tensor temp;

	@Override
	protected void configure(Map<String, Object> properties) {
		if(properties.containsKey("binarize"))
			this.threshold = Float.parseFloat((String) properties.get("binarize"));
	}

	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		temp = TensorOps.sub(temp, original.input, threshold);
		adapted.input = TensorOps.sign(adapted.input, temp);
		TensorOps.abs(temp, adapted.input);
		TensorOps.add(adapted.input, adapted.input, temp);
		TensorOps.div(adapted.input, adapted.input, 2);
		
		adapted.target = original.target.copyInto(adapted.target);
	}

}
