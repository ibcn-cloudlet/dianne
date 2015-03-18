package be.iminds.iot.dianne.things.camera;

import java.awt.Canvas;
import java.awt.Frame;
import java.awt.Graphics2D;
import java.awt.Toolkit;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;

import be.iminds.iot.dianne.tensor.Tensor;

public class CameraCanvas extends Canvas {
	
	static {
		System.setProperty("sun.java2d.transaccel", "True");
		System.setProperty("sun.java2d.opengl", "True");
	}
	
	private Frame frame;
	private BufferStrategy strategy;

	private BufferedImage view;
	private final int width = 28;
	private final int height = 28;
	
	public CameraCanvas(){
		init();
	}
	
	public void render(Tensor t) {
		float[] data = t.get();
		int k = 0;
		for (int j = 0; j < t.size(0); j++) {
			for (int i = 0; i < t.size(1); i++) {
				int v = 255-(int)(data[k++]*255);
				final int r = v;
				final int g = v;
				final int b = v;
				final int a = 255;
				final int col = a << 24 | r << 16 | g << 8 | b;
				this.view.setRGB(i, j, col);
			}
		}
		
		final Graphics2D bkG = (Graphics2D) this.strategy.getDrawGraphics();
		bkG.drawImage(this.view, 0, 0, width, height, null);
		bkG.dispose();
		this.strategy.show();
		Toolkit.getDefaultToolkit().sync();
	}

	private void init(){
		this.setIgnoreRepaint(true);
		
		frame = new Frame("");
		frame.add(this);

		frame.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(final WindowEvent e) {
				dispose();
			}
		});

		frame.setSize(28, 56);
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);

		this.view = new BufferedImage(width, height,
				BufferedImage.TYPE_INT_RGB);
		this.createBufferStrategy(2);

		this.strategy = this.getBufferStrategy();
	}

	public void dispose(){
		frame.setVisible(false);
		frame.dispose();
		
	}
}
