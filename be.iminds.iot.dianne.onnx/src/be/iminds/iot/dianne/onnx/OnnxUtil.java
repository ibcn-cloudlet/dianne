package be.iminds.iot.dianne.onnx;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import be.iminds.iot.dianne.tensor.Tensor;
import onnx.Onnx.TensorProto;
import onnx.Onnx.TensorProto.DataType;

final class OnnxUtil {
	
	private OnnxUtil() {}

	static Tensor toTensor(TensorProto t) {
		int[] dims = new int[t.getDimsCount()];
		for(int i=0;i<dims.length;i++) {
			dims[i] = (int)t.getDims(i);
		}

		float[] data;
		ByteBuffer buffer = t.getRawData().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
		if(t.getDataType() == DataType.FLOAT) {
			data = new float[buffer.capacity()/4];
			for(int i=0;i<data.length;i++) {
				data[i] = buffer.getFloat();
			}
		} else if(t.getDataType() == DataType.INT64) {
			data = new float[buffer.capacity()/8];
			for(int i=0;i<data.length;i++) {
				data[i] = (float)buffer.getLong();
			}
		} else if(t.getDataType() == DataType.INT32) {
			data = new float[buffer.capacity()/4];
			for(int i=0;i<data.length;i++) {
				data[i] = (float)buffer.getInt();
			}
		} else {
			throw new RuntimeException("Unsupported data format "+t.getDataType().toString());
		}
		
		
		return new Tensor(data, dims);
	}
}
