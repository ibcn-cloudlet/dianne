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
 *     Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.dataset.imagenet.training;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.dataset.ImageClassificationDataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The Imagenet dataset
 * 
 * Currently based on the complete LSVRC 2012 training set
 * 
 * @author ejodconi
 *
 */
@Component(
		service={Dataset.class, Ranges.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.ImageNet.training",
		property={"osgi.command.scope=imagenet",
				  "osgi.command.function=ranges"})
public class ImageNetTrainingDataset extends ImageClassificationDataset implements Ranges {

	private int[] noSamplesNodeId;
	private String[] nodeIds;

	/*
	 * @see be.iminds.iot.dianne.dataset.imagenet.training.Ranges#ranges(java.lang.String[])
	 */
	public void ranges(String...nodeIds) {
		System.out.println("Ranges:");
		int[] nodeIndexes;
		if (nodeIds.length==0) {
			nodeIndexes = new int[this.nodeIds.length];
			nodeIds = this.nodeIds;
		}
		else
			nodeIndexes = new int[nodeIds.length];
		
		int j=0;
		List<String> list = Arrays.asList(nodeIds);
		for (int i=0;i<this.nodeIds.length; i++) {
			if (list.contains(this.nodeIds[i]))
				nodeIndexes[j++]=i;
		}
		
		for (int nodeIndex : nodeIndexes) {
			if (this.nodeIds[nodeIndex]!=null) {
				int start=0;
				if (nodeIndex>0)
					start=noSamplesNodeId[nodeIndex-1];
				int end=noSamplesNodeId[nodeIndex]-1;
				System.out.printf("\t%s: %d - %d\t(#samples=%d)\n",this.nodeIds[nodeIndex], start, end, end-start+1);
			}
		}
	}
	
	private int processImageDir(Path dir) {
		int count = 0;
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			int nodeIndex = 0;
			for (Path path : stream) {
				if (Files.isDirectory(path)) {
					count += processImageDir(path);
					noSamplesNodeId[nodeIndex] = count;
					nodeIds[nodeIndex] = path.getFileName().toString();
					nodeIndex++;
				} else if (!Files.isHidden(path)) {
					count++;
				}
			}
		} catch (IOException e) { e.printStackTrace(); }
		return count;
	}
	
	@Override
	protected void init(Map<String, Object> properties) {
		this.name = "ImageNetTraining";
		this.inputDims = null;
		this.targetDims = new int[]{1000};
		this.nodeIds = new String[1000];
		this.noSamplesNodeId = new int[1000];
		this.noSamples = processImageDir(Paths.get(dir + File.separator + "images"));
		this.labelsFile = "classes.txt";
	}

	@Override
	protected String getImageFile(int index) {
		int nodeIndex=0;
		while (index >= noSamplesNodeId[nodeIndex])
			nodeIndex++;
		return String.format("%s/images/%s/%d.JPEG", dir, nodeIds[nodeIndex], index-(nodeIndex!=0 ? noSamplesNodeId[nodeIndex-1] : 0)+1);
	}

	@Override
	protected void readLabels(String file) {
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(dir + File.separator + file)))) {
			Map<String,String> m = new HashMap<>();
			
			String l;
			while ((l = reader.readLine()) != null) {
				String[] s = l.split("\t|,");
				m.put(s[0], s[1]);
			}
			List<String> list = new ArrayList<>();
			for (int i=0; i<nodeIds.length; i++) {
				// only pick first label in case of multiple definitions
				l = m.get(nodeIds[i]);
				if (l!= null){
					int comma = l.indexOf(",");
					if (comma > 0) {
						l = l.substring(0, comma);
					}
					list.add(l);
				}
			}
			labels = list.toArray(new String[list.size()]);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void readTargets(String file) {}
	@Override
	public Tensor getTargetSample(Tensor t, int index) {
		if(t == null)
			t = new Tensor(targetSize);
		t.fill(0.0f);
		int nodeIndex=0;
		while (index >= noSamplesNodeId[nodeIndex])
			nodeIndex++;
		t.set(1.0f, nodeIndex);
		return t;
	}
}
