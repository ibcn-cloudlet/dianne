package be.iminds.iot.dianne.rl.experience;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.NativeTensorLoader;
import be.iminds.iot.dianne.tensor.Tensor;
import junit.framework.Assert;

@RunWith(Parameterized.class)
public class ExperiencePoolTest {

	@Parameters()
    public static Collection<Object[]> getParameters() {
      return Arrays.asList(new Object[][] {
        { new FileExperiencePool(), null },
        { new FileExperiencePool(), "200" },
        { new MemoryExperiencePool(), null}
      });
    }
	
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
	
    public ExperiencePoolTest(ExperiencePool p, Object bufferSize) throws Exception {
    	this.pool = p.getClass().newInstance();
    	
    	Map<String, Object> config = new HashMap<>();
		config.put("name", "Test");
		config.put("dir", "generated");
		config.put("stateDims", new String[]{"4"});
		config.put("actionDims", new String[]{"2"});
		config.put("maxSize", "12");
		if(bufferSize!=null){
			config.put("sizePerBuffer", bufferSize);
		}
		
		pool.getClass().getMethod("activate", Map.class).invoke(pool, config);
    }
	
	@Test
	public void testExperiencePoolInit() throws Exception {
		Assert.assertEquals("Test", pool.getName());
		Assert.assertEquals(0, pool.size());
		Assert.assertEquals(0, pool.sequences());
	}
	
	@Test
	public void testExperiencePoolAdd() throws Exception {
		Assert.assertEquals(0, pool.size());
		Assert.assertEquals(0, pool.sequences());
		
		// add first sequence
		List<ExperiencePoolSample> sequence = new ArrayList<>();
		sequence.add(new ExperiencePoolSample(s0, a0, 0, s1));
		sequence.add(new ExperiencePoolSample(s1, a1, 0, s2));
		sequence.add(new ExperiencePoolSample(s2, a2, 0, s3));
		sequence.add(new ExperiencePoolSample(s3, a3, 1, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence, 4));
		
		Assert.assertEquals(4, pool.size());
		Assert.assertEquals(1, pool.sequences());
		
		ExperiencePoolSample s = pool.getSample(1);
		Assert.assertEquals(s1, s.getState());
		Assert.assertEquals(a1, s.getAction());
		Assert.assertEquals(0.0f, s.getScalarReward());
		Assert.assertEquals(s2, s.getNextState());
		Assert.assertEquals(false, s.isTerminal());
		
		ExperiencePoolSample end = pool.getSample(3);
		Assert.assertEquals(s3, end.getState());
		Assert.assertEquals(a3, end.getAction());
		Assert.assertEquals(1.0f, end.getScalarReward());
		Tensor nan = end.getNextState();
		for(float n : nan.get()){
			Assert.assertEquals(0.0f, n);
		}
		Assert.assertEquals(true, end.isTerminal());	
		
		
		// add second sequence
		List<ExperiencePoolSample> sequence2 = new ArrayList<>();
		sequence2.add(new ExperiencePoolSample(s0, a0, 0, s1));
		sequence2.add(new ExperiencePoolSample(s1, a1, 0, s2));
		sequence2.add(new ExperiencePoolSample(s2, a2, 0, s3));
		sequence2.add(new ExperiencePoolSample(s3, a3, 0, s4));
		sequence2.add(new ExperiencePoolSample(s4, a4, 0, s5));
		sequence2.add(new ExperiencePoolSample(s5, a5, 0, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence2, 6));
		
		Assert.assertEquals(10, pool.size());
		Assert.assertEquals(2, pool.sequences());
		
		s = pool.getSample(1);
		Assert.assertEquals(s1, s.getState());
		Assert.assertEquals(a1, s.getAction());
		Assert.assertEquals(0.0f, s.getScalarReward());
		Assert.assertEquals(s2, s.getNextState());
		Assert.assertEquals(false, s.isTerminal());
		
		s = pool.getSample(4);
		Assert.assertEquals(s0, s.getState());
		Assert.assertEquals(a0, s.getAction());
		Assert.assertEquals(0.0f, s.getScalarReward());
		Assert.assertEquals(s1, s.getNextState());
		Assert.assertEquals(false, s.isTerminal());
		
		end = pool.getSample(9);
		Assert.assertEquals(s5, end.getState());
		Assert.assertEquals(a5, end.getAction());
		Assert.assertEquals(0.0f, end.getScalarReward());
		Assert.assertEquals(sEnd, end.getNextState());

		Assert.assertEquals(true, end.isTerminal());	
		
		
		// test get sequence
		Sequence<ExperiencePoolSample> retrieved = pool.getSequence(1);
		int i = 0;
		for(ExperiencePoolSample r : retrieved){
			ExperiencePoolSample expected = sequence2.get(i++);
			Assert.assertEquals(expected.getState(), r.getState());
			Assert.assertEquals(expected.getAction(), r.getAction());
			Assert.assertEquals(expected.getScalarReward(), r.getScalarReward());
			Assert.assertEquals(expected.isTerminal(), r.isTerminal());
			if(expected.isTerminal()){
				Assert.assertEquals(sEnd, r.getNextState());
			} else {
				Assert.assertEquals(expected.getNextState(), r.getNextState());
			}
		}
	}
	
	@Test
	public void testExperiencePoolCycle() throws Exception {
		Assert.assertEquals(0, pool.size());
		Assert.assertEquals(0, pool.sequences());
		
		// first sequence
		List<ExperiencePoolSample> sequence = new ArrayList<>();
		sequence.add(new ExperiencePoolSample(s0, a0, 0, s1));
		sequence.add(new ExperiencePoolSample(s1, a1, 0, s2));
		sequence.add(new ExperiencePoolSample(s2, a2, 0, s3));
		sequence.add(new ExperiencePoolSample(s3, a3, 1, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence, 4));
		
		Assert.assertEquals(4, pool.size());
		Assert.assertEquals(1, pool.sequences());

		// second sequence
		List<ExperiencePoolSample> sequence2 = new ArrayList<>();
		// use s1 here as start so we can check whether we correctly cycled
		// this sample should be sample 0 after cycling
		sequence2.add(new ExperiencePoolSample(s1, a0, 0, s1));
		sequence2.add(new ExperiencePoolSample(s1, a1, 0, s2));
		sequence2.add(new ExperiencePoolSample(s2, a2, 0, s3));
		sequence2.add(new ExperiencePoolSample(s3, a3, 0, s4));
		sequence2.add(new ExperiencePoolSample(s4, a4, 0, s5));
		sequence2.add(new ExperiencePoolSample(s5, a5, 0, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence2, 6));
		
		Assert.assertEquals(10, pool.size());
		Assert.assertEquals(2, pool.sequences());
		
		// third sequence - should cycle
		List<ExperiencePoolSample> sequence3 = new ArrayList<>();
		sequence3.add(new ExperiencePoolSample(s0, a0, 1, s1));
		sequence3.add(new ExperiencePoolSample(s1, a1, 0, s2));
		sequence3.add(new ExperiencePoolSample(s2, a2, 0, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence3, 3));
		
		Assert.assertEquals(9, pool.size());
		Assert.assertEquals(2, pool.sequences());	
		
		// this should be first sample of sequence2
		ExperiencePoolSample s = pool.getSample(0);
		Assert.assertEquals(s1, s.getState());
		Assert.assertEquals(a0, s.getAction());
		Assert.assertEquals(0.0f, s.getScalarReward());
		Assert.assertEquals(s1, s.getNextState());
		Assert.assertEquals(false, s.isTerminal());
		
		// this should be first sample of sequence3
		s = pool.getSample(6);
		Assert.assertEquals(s0, s.getState());
		Assert.assertEquals(a0, s.getAction());
		Assert.assertEquals(1.0f, s.getScalarReward());
		Assert.assertEquals(s1, s.getNextState());
		Assert.assertEquals(false, s.isTerminal());
		
		// now add additional instances of sequence3
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence3, 3));

		Assert.assertEquals(12, pool.size());
		Assert.assertEquals(3, pool.sequences());
		
		// and cycle some more
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence2, 6));
		Assert.assertEquals(12, pool.size());
		Assert.assertEquals(3, pool.sequences());
		
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence2, 6));
		Assert.assertEquals(12, pool.size());
		Assert.assertEquals(2, pool.sequences());
		
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence3, 3));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence3, 3));
		Assert.assertEquals(12, pool.size());
		Assert.assertEquals(3, pool.sequences());
		
	}
	
	@Test
	public void testSequenceTooLarge(){
		Assert.assertEquals(0, pool.size());
		Assert.assertEquals(0, pool.sequences());
		
		List<ExperiencePoolSample> sequence = new ArrayList<>();
		for(int i=0;i<20;i++)
			sequence.add(new ExperiencePoolSample(s0, a0, 0, s0));
		
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence, 20));
		
		Assert.assertEquals(0, pool.size());
		Assert.assertEquals(0, pool.sequences());
	}
	
	@Test
	public void testBatch(){
		Assert.assertEquals(0, pool.size());
		Assert.assertEquals(0, pool.sequences());
		
		// first sequence
		List<ExperiencePoolSample> sequence = new ArrayList<>();
		sequence.add(new ExperiencePoolSample(s0, a0, 0, s1));
		sequence.add(new ExperiencePoolSample(s1, a1, 0.1f, s2));
		sequence.add(new ExperiencePoolSample(s2, a2, 0.2f, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence, 3));

		// second sequence
		List<ExperiencePoolSample> sequence2 = new ArrayList<>();
		sequence2.add(new ExperiencePoolSample(s0, a0, 0.3f, s1));
		sequence2.add(new ExperiencePoolSample(s1, a1, 0.4f, s2));
		sequence2.add(new ExperiencePoolSample(s2, a2, 0.5f, s3));
		sequence2.add(new ExperiencePoolSample(s3, a3, 0.6f, s4));
		sequence2.add(new ExperiencePoolSample(s4, a4, 0.7f, s5));
		sequence2.add(new ExperiencePoolSample(s5, a5, 0.8f, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence2, 6));
		
		ExperiencePoolBatch batch = pool.getBatch(0, 1, 3, 8, 5);
		Assert.assertEquals(5, batch.getSize());
		System.out.println(batch);
		Tensor bState = new Tensor(new float[]{0.0f, 1.0f, 2.0f, 3.0f,
											  1.0f, 2.0f, 3.0f, 4.0f, 
											  0.0f, 1.0f, 2.0f, 3.0f,
											  5.0f, 6.0f, 7.0f, 8.0f,
											  2.0f, 3.0f, 4.0f, 5.0f}, 5, 4);
		Assert.assertEquals(bState, batch.getState());
	
		Tensor bAction = new Tensor(new float[]{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 1.0f, 1.0f, 0.0f}, 5, 2);
		Assert.assertEquals(bAction, batch.getAction());

		Tensor bReward = new Tensor(new float[]{0.0f, 0.1f, 0.3f, 0.8f, 0.5f}, 5, 1);
		Assert.assertEquals(bReward, batch.getReward());
		
		Tensor bNextState = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 
												2.0f, 3.0f, 4.0f, 5.0f,
												1.0f, 2.0f, 3.0f, 4.0f,
												0.0f, 0.0f, 0.0f, 0.0f,
												3.0f, 4.0f, 5.0f, 6.0f}, 5, 4);
		Assert.assertEquals(bNextState, batch.getNextState());

		Tensor bTerminal = new Tensor(new float[]{1.0f, 1.0f, 1.0f, 0.0f, 1.0f}, 5, 1);
		Assert.assertEquals(bTerminal, batch.getTerminal());

	}
	
	@Test
	public void testBatchedSequence(){
		Assert.assertEquals(0, pool.size());
		Assert.assertEquals(0, pool.sequences());
		
		// first sequence
		List<ExperiencePoolSample> sequence = new ArrayList<>();
		sequence.add(new ExperiencePoolSample(s0, a0, 0, s1));
		sequence.add(new ExperiencePoolSample(s1, a1, 0.1f, s2));
		sequence.add(new ExperiencePoolSample(s2, a2, 0.2f, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence, 3));

		// second sequence
		List<ExperiencePoolSample> sequence2 = new ArrayList<>();
		sequence2.add(new ExperiencePoolSample(s0, a0, 0.3f, s1));
		sequence2.add(new ExperiencePoolSample(s1, a1, 0.4f, s2));
		sequence2.add(new ExperiencePoolSample(s2, a2, 0.5f, s3));
		sequence2.add(new ExperiencePoolSample(s3, a3, 0.6f, s4));
		sequence2.add(new ExperiencePoolSample(s4, a4, 0.7f, s5));
		sequence2.add(new ExperiencePoolSample(s5, a5, 0.8f, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence2, 6));
		
		Sequence<ExperiencePoolBatch> batchedSequence = pool.getBatchedSequence(new int[]{0, 1}, new int[]{0, 1}, 3);
		Assert.assertEquals(3, batchedSequence.size);
		ExperiencePoolBatch b0 = batchedSequence.get(0);
		Assert.assertEquals(s0, b0.getState(0));
		Assert.assertEquals(s1, b0.getState(1));

		ExperiencePoolBatch b1 = batchedSequence.get(1);
		Assert.assertEquals(s1, b1.getState(0));
		Assert.assertEquals(s2, b1.getState(1));
		
		
		batchedSequence = pool.getBatchedSequence(batchedSequence, new int[]{0, 1}, new int[]{1, 1}, 3);
		Assert.assertEquals(2, batchedSequence.size);
		b0 = batchedSequence.get(0);
		Assert.assertEquals(s1, b0.getState(0));
		Assert.assertEquals(s1, b0.getState(1));

		b1 = batchedSequence.get(1);
		Assert.assertEquals(s2, b1.getState(0));
		Assert.assertEquals(s2, b1.getState(1));
		
	}
	
	
	@Test
	public void testExperiencePoolGetSubSequence() throws Exception {
		Assert.assertEquals(0, pool.size());
		Assert.assertEquals(0, pool.sequences());
		
		// add first sequence
		List<ExperiencePoolSample> sequence = new ArrayList<>();
		sequence.add(new ExperiencePoolSample(s0, a0, 0, s1));
		sequence.add(new ExperiencePoolSample(s1, a1, 0, s2));
		sequence.add(new ExperiencePoolSample(s2, a2, 0, s3));
		sequence.add(new ExperiencePoolSample(s3, a3, 1, null));
		pool.addSequence(new Sequence<ExperiencePoolSample>(sequence, 4));
		
		Assert.assertEquals(4, pool.size());
		Assert.assertEquals(1, pool.sequences());
		
		// test get sequence
		Sequence<ExperiencePoolSample> retrieved = pool.getSequence(0, 1, -1);
		Assert.assertEquals(3, retrieved.size());
		int i = 1;
		for(ExperiencePoolSample r : retrieved){
			ExperiencePoolSample expected = sequence.get(i++);
			Assert.assertEquals(expected.getState(), r.getState());
			Assert.assertEquals(expected.getAction(), r.getAction());
			Assert.assertEquals(expected.getScalarReward(), r.getScalarReward());
			Assert.assertEquals(expected.isTerminal(), r.isTerminal());
			if(expected.isTerminal()){
				Assert.assertEquals(sEnd, r.getNextState());
			} else {
				Assert.assertEquals(expected.getNextState(), r.getNextState());
			}
		}
		
		retrieved = pool.getSequence(0, 1, 2);
		Assert.assertEquals(2, retrieved.size());
		i = 1;
		for(ExperiencePoolSample r : retrieved){
			ExperiencePoolSample expected = sequence.get(i++);
			Assert.assertEquals(expected.getState(), r.getState());
			Assert.assertEquals(expected.getAction(), r.getAction());
			Assert.assertEquals(expected.getScalarReward(), r.getScalarReward());
			Assert.assertEquals(expected.isTerminal(), r.isTerminal());
			if(expected.isTerminal()){
				Assert.assertEquals(sEnd, r.getNextState());
			} else {
				Assert.assertEquals(expected.getNextState(), r.getNextState());
			}
		}
		
		retrieved = pool.getSequence(0, 1, 4);
		Assert.assertEquals(3, retrieved.size());
		i = 1;
		for(ExperiencePoolSample r : retrieved){
			ExperiencePoolSample expected = sequence.get(i++);
			Assert.assertEquals(expected.getState(), r.getState());
			Assert.assertEquals(expected.getAction(), r.getAction());
			Assert.assertEquals(expected.getScalarReward(), r.getScalarReward());
			Assert.assertEquals(expected.isTerminal(), r.isTerminal());
			if(expected.isTerminal()){
				Assert.assertEquals(sEnd, r.getNextState());
			} else {
				Assert.assertEquals(expected.getNextState(), r.getNextState());
			}
		}
	}
}
