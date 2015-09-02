package be.iminds.iot.dianne.tensor.impl.th;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

@Component(service=Serializer.class, property={"aiolos.export=false",
	"kryo.serializer.class=be.iminds.iot.dianne.tensor.impl.th.THTensor",
	"kryo.serializer.id=100"})
public class THTensorSerializer extends Serializer<Tensor> {

	private TensorFactory factory;
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}

	@Override
	public Tensor read(Kryo kryo, Input input, Class<Tensor> tensor) {
		int noDims = input.readInt();
		int[] dims = input.readInts(noDims);
		int length = input.readInt();
		float[] data = input.readFloats(length);
		return factory.createTensor(data, dims);
	}

	@Override
	public void write(Kryo kryo, Output output, Tensor tensor) {
		output.writeInt(tensor.dims().length);
		output.writeInts(tensor.dims());
		output.writeInt(tensor.size());
		output.writeFloats(tensor.get());
	}

}
