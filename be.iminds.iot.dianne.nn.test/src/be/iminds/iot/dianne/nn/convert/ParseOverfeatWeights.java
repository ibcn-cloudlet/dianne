package be.iminds.iot.dianne.nn.convert;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ParseOverfeatWeights {

	public static void main(String[] args) throws IOException {
		
		// the weights file of overfeat as available on 
		// http://cilvr.nyu.edu/lib/exe/fetch.php?media=overfeat:overfeat-weights.tgz
		//
		// currently we only load the 'fast' variant
		String overfeat_fast = "/home/tverbele/Desktop/overfeat/data/default/net_weight_0";
		
		String normalization = "58fe71c5-13ee-2fb1-a05f-8e621e991389";
		String[] layers = {
			"d8b16517-dd36-0fa7-2dea-89d95e234552", // 96x11x11
			"3b425e33-2247-a498-2524-95c11677400a", // 256x5x5
			"c7b77bb8-400e-4c56-3bea-5a532e0729b7", // 512x3x3
			"ec9445bc-f791-ebb7-d6ce-892241fa9f9e", // 1024x3x3
			"363600ee-d707-2e63-cb0a-37da0914b853", // 1024x3x3
			"745eb02f-9de2-2676-3446-78617362f67b", // 36864x3072
			"2dc69524-ca55-7051-840a-90fa6847a3d5", // 3072x4096
			"355a0d69-b7f8-1a67-7002-70facd17f908" // 4096x1000
		};
		
		int[] sizes = {
			96*3*11*11+96,
			256*96*5*5+256,
			512*256*3*3+512,
			1024*512*3*3+1024,
			1024*1024*3*3+1024,
			3072*(36864+1),
			4096*(3072+1),
			1000*(4096+1)
		};
		
		File f = new File(overfeat_fast);
		
		// load all weights and create corresponding files for dianne config
		InputStream i = new FileInputStream(f);
		for(int layer = 0;layer<sizes.length;layer++){
			int size = sizes[layer];
			float[] weights = new float[size];
			for(int k=0;k<size;k++){
				weights[k] = readFloat(i);
			}
			
			DataOutputStream out = new DataOutputStream(new FileOutputStream(new File(layers[layer])));
			out.writeInt(size);
			for(int k=0;k<size;k++){
				out.writeFloat(weights[k]);
			}
			out.flush();
			out.close();
		}
		i.close();
		
		// also add normalization parameters
		// TODO these are hard coded from the source code
		float mean = 118.380948f/255;
		float std = 61.896913f/255;
		DataOutputStream out = new DataOutputStream(new FileOutputStream(new File(normalization)));
		out.writeInt(2);
		out.writeFloat(mean);
		out.writeFloat(std);
		out.flush();
		out.close();
	}
	
	private static float readFloat(InputStream i) throws IOException{
		byte[] bytes = new byte[4];
		i.read(bytes, 0, 4);
		float f = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getFloat();
		return f;
	}
}
