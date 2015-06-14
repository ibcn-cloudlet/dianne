package be.iminds.iot.dianne.nn.convert;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ParseOverfeatWeights {

	private static class OverfeatConfig {
		String weights; // weights file
		String[] layers; // layer UUIDs in order
		int[] sizes; // parameters size for each layer
		
		// normalization is treated somewhat special
		String normalization; // normalization module UUID
		float mean; // mean for normalization
		float stdev; // stdev for normalization
	}
	
	public static void main(String[] args) throws Exception {
		
		// the weights file of overfeat as available on 
		// http://cilvr.nyu.edu/lib/exe/fetch.php?media=overfeat:overfeat-weights.tgz
		
		// the 'fast' variant
		OverfeatConfig fast = new OverfeatConfig();
		fast.weights = "/home/tverbele/Desktop/overfeat/data/default/net_weight_0";
		fast.layers = new String[]{
			"d8b16517-dd36-0fa7-2dea-89d95e234552", // 96x11x11
			"3b425e33-2247-a498-2524-95c11677400a", // 256x5x5
			"c7b77bb8-400e-4c56-3bea-5a532e0729b7", // 512x3x3
			"ec9445bc-f791-ebb7-d6ce-892241fa9f9e", // 1024x3x3
			"363600ee-d707-2e63-cb0a-37da0914b853", // 1024x3x3
			"745eb02f-9de2-2676-3446-78617362f67b", // 36864x3072
			"2dc69524-ca55-7051-840a-90fa6847a3d5", // 3072x4096
			"355a0d69-b7f8-1a67-7002-70facd17f908" // 4096x1000
		};
		fast.sizes = new int[]{
			96*3*11*11+96,
			256*96*5*5+256,
			512*256*3*3+512,
			1024*512*3*3+1024,
			1024*1024*3*3+1024,
			3072*(36864+1),
			4096*(3072+1),
			1000*(4096+1)
		};
		fast.normalization = "58fe71c5-13ee-2fb1-a05f-8e621e991389";
		fast.mean = 118.380948f/255;
		fast.stdev = 61.896913f/255;

		
		// the 'accurate' variant
		OverfeatConfig accurate = new OverfeatConfig();
		accurate.weights = "/home/tverbele/Desktop/overfeat/data/default/net_weight_1";
		accurate.layers = new String[]{
			"bb3a70c4-91ec-b3c6-7d3b-72f339e06605", // 96x7x7
			"f6d11058-764c-77cd-f7e5-3a22a4bb8891", // 256x7x7
			"7e0b28eb-eb8f-5243-6800-f8face2b29b0", // 512x3x3
			"6fbce2c0-dd1d-cf36-8353-c40c62cdbeca", // 512x3x3
			"7aaac5b3-4022-f38b-a759-cfeecb249fd7", // 1024x3x3
			"c5773457-2ff4-a144-7afb-ce254c7021a5", // 1024x3x3
			"4aaec81c-5015-5830-7065-3d8d5b630956", // 25600x4096
			"8cbd86d1-d2f4-6e0f-2bb3-ba309a041e08", // 4096x4096
			"ca89fcf5-efeb-2461-bedf-adb4abb8ff63" // 4096x1000
		};
		accurate.sizes = new int[]{
			96*3*7*7+96,
			256*96*7*7+256,
			512*256*3*3+512,
			512*512*3*3+512,
			1024*512*3*3+1024,
			1024*1024*3*3+1024,
			4096*(25600+1),
			4096*(4096+1),
			1000*(4096+1)
		};
		accurate.normalization = "5a46b6cf-2525-bb71-6c8e-5e8e534287e0";
		accurate.mean = 118.380948f/255;
		accurate.stdev = 61.896913f/255;
		
		//generateModuleWeights(fast);
		generateModuleWeights(accurate);
	}
	
	private static void generateModuleWeights(OverfeatConfig config) throws Exception {
		File f = new File(config.weights);
		
		// load all weights and create corresponding files for dianne config
		InputStream i = new BufferedInputStream(new FileInputStream(f));
		for(int layer = 0;layer<config.sizes.length;layer++){
			System.out.println("Writing weights for module "+config.layers[layer]);
			int size = config.sizes[layer];
			DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(new File(config.layers[layer]))));
			out.writeInt(size);
			for(int k=0;k<size;k++){
				out.writeFloat(readFloat(i));
			}
			out.flush();
			out.close();
		}
		i.close();
		
		System.out.println("Writing parameters for normalization layer");
		// also add normalization parameters
		DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(new File(config.normalization))));
		out.writeInt(2);
		out.writeFloat(config.mean);
		out.writeFloat(config.stdev);
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
