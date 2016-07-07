package be.iminds.iot.dianne.dataset.kaggle;

import java.io.File;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.dataset.ImageSegmentationDataset;

/**
 * Kaggle ultrasound nerve segmentation task dataset
 * 
 * https://www.kaggle.com/c/ultrasound-nerve-segmentation/
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.UltrasoundNerves")
public class UltrasoundNervesDataset extends ImageSegmentationDataset {

	@Override
	protected String getImageFile(int index) {
		int i = index / 120;
		int j = index % 120;
		return dir+File.separator+ "train" + File.separator
				 + i +"_"+j+".tif";
	}

	@Override
	protected String getMaskFile(int index) {
		int i = index / 120;
		int j = index % 120;
		return dir+File.separator+ "train" + File.separator
				 + i +"_"+j+"_mask.tif";
	}

	@Override
	protected void init(Map<String, Object> properties) {
		this.name = "UltrasoundNerves";
		this.inputDims = new int[]{1,420,580};
		this.noSamples = 47*120;
	}
}
