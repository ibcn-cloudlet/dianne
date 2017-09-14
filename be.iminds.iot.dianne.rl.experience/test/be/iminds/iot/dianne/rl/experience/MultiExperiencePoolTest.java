package be.iminds.iot.dianne.rl.experience;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.BeforeClass;
import org.junit.Test;

import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSequence;
import be.iminds.iot.dianne.rl.experience.adapters.MultiExperiencePoolAdapter;
import be.iminds.iot.dianne.tensor.NativeTensorLoader;
import be.iminds.iot.dianne.tensor.Tensor;
import junit.framework.Assert;


public class MultiExperiencePoolTest {
	
	protected ExperiencePool pool;
	
	protected Tensor s0 = new Tensor(new float[]{0, 1, 2, 3}, 4);
	protected Tensor a0 = new Tensor(new float[]{0, 0}, 2);
	protected Tensor s1 = new Tensor(new float[]{1, 2, 3, 4}, 4);
	protected Tensor a1 = new Tensor(new float[]{0, 1}, 2);
	protected Tensor s2 = new Tensor(new float[]{2, 3, 4, 5}, 4);
	protected Tensor a2 = new Tensor(new float[]{1, 0}, 2);
	protected Tensor s3 = new Tensor(new float[]{3, 4, 5, 6}, 4);
	protected Tensor a3 = new Tensor(new float[]{1, 1}, 2);
	protected Tensor s4 = new Tensor(new float[]{4, 5, 6, 7}, 4);
	protected Tensor a4 = new Tensor(new float[]{2, 0}, 2);
	protected Tensor s5 = new Tensor(new float[]{5, 6, 7, 8}, 4);
	protected Tensor a5 = new Tensor(new float[]{2, 1}, 2);
	protected Tensor sEnd = new Tensor(new float[]{0, 0, 0, 0}, 4);
	
	@BeforeClass
	public static void setup() {
		NativeTensorLoader loader = new NativeTensorLoader();
		loader.activate(null);
	}
	
    public MultiExperiencePoolTest() throws Exception {
    	
    	Map<String, Object> config = new HashMap<>();
		config.put("name", "Test1");
		config.put("dir", "generated");
		config.put("stateDims", new String[]{"4"});
		config.put("actionDims", new String[]{"2"});
		config.put("maxSize", "12");
		
    	ExperiencePool pool1 = new MemoryExperiencePool();
		pool1.getClass().getMethod("activate", Map.class).invoke(pool1, config);
		
    	ExperiencePool pool2 = new MemoryExperiencePool();
    	config.put("name", "Test2");
		pool2.getClass().getMethod("activate", Map.class).invoke(pool2, config);
	
		List<ExperiencePoolSample> sequence = new ArrayList<>();
		sequence.add(new ExperiencePoolSample(s0, a0, 1, s1));
		sequence.add(new ExperiencePoolSample(s1, a1, 1, s2));
		sequence.add(new ExperiencePoolSample(s2, a2, 1, s3));
		sequence.add(new ExperiencePoolSample(s3, a3, 1, s4));
		pool1.addSequence(new Sequence<ExperiencePoolSample>(sequence, 4));
		
		List<ExperiencePoolSample> sequence2 = new ArrayList<>();
		sequence2.add(new ExperiencePoolSample(s1, a0, 2, s1));
		sequence2.add(new ExperiencePoolSample(s1, a1, 2, s2));
		sequence2.add(new ExperiencePoolSample(s2, a2, 2, s3));
		sequence2.add(new ExperiencePoolSample(s3, a3, 2, s4));
		sequence2.add(new ExperiencePoolSample(s4, a4, 2, s5));
		pool1.addSequence(new Sequence<ExperiencePoolSample>(sequence2, 5));
		
		List<ExperiencePoolSample> sequence3 = new ArrayList<>();
		sequence3.add(new ExperiencePoolSample(s0, a0, 3, s1));
		sequence3.add(new ExperiencePoolSample(s1, a1, 3, s2));
		sequence3.add(new ExperiencePoolSample(s2, a2, 3, s3));
		sequence3.add(new ExperiencePoolSample(s3, a3, 3, s4));
		pool2.addSequence(new Sequence<ExperiencePoolSample>(sequence3, 4));
		
		List<ExperiencePoolSample> sequence4 = new ArrayList<>();
		sequence4.add(new ExperiencePoolSample(s0, a0, 4, s1));
		sequence4.add(new ExperiencePoolSample(s1, a1, 4, s2));
		sequence4.add(new ExperiencePoolSample(s2, a2, 4, s3));
		sequence4.add(new ExperiencePoolSample(s3, a3, 4, s4));
		pool2.addSequence(new Sequence<ExperiencePoolSample>(sequence4, 4));
		
		pool = new MultiExperiencePoolAdapter();
		config.put("name", "Multi");
		pool.getClass().getMethod("activate", Map.class).invoke(pool, config);
		pool.getClass().getMethod("addDataset", ExperiencePool.class).invoke(pool, pool1);
		pool.getClass().getMethod("addDataset", ExperiencePool.class).invoke(pool, pool2);
    }
	

    @Test
    public void testSize(){
		Assert.assertEquals(17, pool.size());
    }
    
    @Test
    public void testSequences(){
		Assert.assertEquals(4, pool.sequences());
    }
    
    @Test
    public void testSequenceLength(){
		Assert.assertEquals(4, pool.sequenceLength(0));
		Assert.assertEquals(5, pool.sequenceLength(1));
		Assert.assertEquals(4, pool.sequenceLength(2));
		Assert.assertEquals(4, pool.sequenceLength(3));
    }
    
	@Test
	public void testSample(){
		Assert.assertEquals(new ExperiencePoolSample(s2, a2, 1, s3), pool.getSample(2));
		Assert.assertEquals(new ExperiencePoolSample(s2, a2, 2, s3), pool.getSample(6));
		Assert.assertEquals(new ExperiencePoolSample(s0, a0, 3, s1), pool.getSample(9));
		Assert.assertEquals(new ExperiencePoolSample(s3, a3, 4, s4), pool.getSample(16));
	}
	
	@Test
	public void testBatch(){
		ExperiencePoolBatch b = new ExperiencePoolBatch(4, new int[]{4}, new int[]{2});
		new ExperiencePoolSample(s2, a2, 1, s3).copyInto(b.getSample(0));
		new ExperiencePoolSample(s2, a2, 2, s3).copyInto(b.getSample(1));
		new ExperiencePoolSample(s0, a0, 3, s1).copyInto(b.getSample(2));
		new ExperiencePoolSample(s3, a3, 4, s4).copyInto(b.getSample(3));
		Assert.assertEquals(b, pool.getBatch(2,6,9,16));
	}
	
	@Test
	public void testSequence(){
		List<ExperiencePoolSample> sequence = new ArrayList<>();
		sequence.add(new ExperiencePoolSample(s1, a0, 2, s1));
		sequence.add(new ExperiencePoolSample(s1, a1, 2, s2));
		sequence.add(new ExperiencePoolSample(s2, a2, 2, s3));
		sequence.add(new ExperiencePoolSample(s3, a3, 2, s4));
		sequence.add(new ExperiencePoolSample(s4, a4, 2, s5));
		Assert.assertEquals(new ExperiencePoolSequence(sequence, 5), pool.getSequence(1));

		List<ExperiencePoolSample> sequence2 = new ArrayList<>();
		sequence2.add(new ExperiencePoolSample(s1, a1, 3, s2));
		sequence2.add(new ExperiencePoolSample(s2, a2, 3, s3));
		sequence2.add(new ExperiencePoolSample(s3, a3, 3, s4));
		Assert.assertEquals(new ExperiencePoolSequence(sequence2, 3), pool.getSequence(2, 1, 3));
	}
	
	@Test
	public void testBatchedSequence(){
		ExperiencePoolBatch b0 = new ExperiencePoolBatch(2, new int[]{4}, new int[]{2});
		new ExperiencePoolSample(s1, a0, 2, s1).copyInto(b0.getSample(0));
		new ExperiencePoolSample(s1, a1, 3, s2).copyInto(b0.getSample(1));

		ExperiencePoolBatch b1 = new ExperiencePoolBatch(2, new int[]{4}, new int[]{2});
		new ExperiencePoolSample(s1, a1, 2, s2).copyInto(b1.getSample(0));
		new ExperiencePoolSample(s2, a2, 3, s3).copyInto(b1.getSample(1));
		
		ExperiencePoolBatch b2 = new ExperiencePoolBatch(2, new int[]{4}, new int[]{2});
		new ExperiencePoolSample(s2, a2, 2, s3).copyInto(b2.getSample(0));
		new ExperiencePoolSample(s3, a3, 3, s4).copyInto(b2.getSample(1));
		
		Sequence<ExperiencePoolBatch> s = pool.getBatchedSequence(new int[]{1, 2}, new int[]{0, 1}, 3);
		
		Assert.assertEquals(3, s.size());
		Assert.assertEquals(b0, s.get(0));
		Assert.assertEquals(b1, s.get(1));
		Assert.assertEquals(b2, s.get(2));
	}

}
