package be.iminds.iot.dianne.nn.util.strategy;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

import org.osgi.framework.Bundle;
import org.osgi.framework.BundleContext;
import org.osgi.framework.wiring.BundleWire;
import org.osgi.framework.wiring.BundleWiring;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.util.StrategyFactory;

@Component(property={"aiolos.proxy=false"})
public class StrategyFactoryImpl<T> implements StrategyFactory<T>{

	private BundleContext context;
	private BundleWiring wiring;
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
		this.wiring = context.getBundle().adapt(BundleWiring.class);
	}
	
	@Override
	public T create(String strategy) {
		if(strategy.contains("class")){
			try {
				return createFromSource(strategy);
			} catch(Exception e){
				e.printStackTrace();
				return null;
			}
		}
		
		Class c;
		try {
			if(strategy.contains(".")){
				// fully qualified class name, load directly
				c = wiring.getClassLoader().loadClass(strategy);

			} else {
				// search resource in this bundles wiring
				List<URL> urls = new ArrayList<>();
				
				urls.addAll(wiring.findEntries("/", strategy+".class", BundleWiring.FINDENTRIES_RECURSE));
				for(BundleWire w : wiring.getRequiredWires(null)){
					List<URL> u = w.getProvider().getWiring().findEntries("/", strategy+".class", BundleWiring.FINDENTRIES_RECURSE);
					urls.addAll(u);
				}
				
				if(urls.size() == 0){
					System.err.println("Strategy "+strategy+" not found");
					return null;
				}
				
				String u = urls.get(0).toString().substring(9);
				u = u.substring(u.indexOf("/")+1, u.length()-6);
				u = u.replaceAll("/", ".");
		
				c = this.getClass().getClassLoader().loadClass(u);
			}
			
			return (T) c.newInstance();
		} catch (Throwable e) {
			e.printStackTrace();
		}
		
		return null;
	}

	
	private T createFromSource(String source) throws Exception {
		// fetch package
		String pkg = "";
		int pkgStart = source.indexOf("package");
		if(pkgStart >= 0){
			int pkgEnd = source.indexOf(";", pkgStart);
			pkg = source.substring(pkgStart+7, pkgEnd).trim();
		} else {
			// no package specified
			// add a package since OSGi does not like the default package
			source = "package pkg;"+source;
			pkg = "pkg";
		}

		// fetch the class name
		int start = source.indexOf("public class");
		int end = source.indexOf("{", start);
		String className = source.substring(start+12, end);
		className = className.trim();
		if(className.contains(" ")){
			className = className.substring(0, className.indexOf(' '));
		}
		
		String pkgDir = pkg.replaceAll("\\.", "/");

		// fully qualified name
		String fqn = className;
		if(pkg.length() > 0)
			fqn = pkg+"."+className;
		
		// write source file to temp folder
		File root = new File(System.getProperty("java.io.tmpdir")+File.separator+"strategy");
		root.mkdirs();
		File dir = new File(root.getAbsolutePath()+File.separator+pkgDir);
		dir.mkdirs();
		File sourceFile = new File(dir.getAbsolutePath()+File.separator+className+".java");
		Files.write(sourceFile.toPath(), source.getBytes(StandardCharsets.UTF_8));
		
		// compile source file
		JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();

		// build classpath
		String classpath = "";
		// also put all bundles on the compiler classpath
		for(Bundle b : context.getBundles()){
			classpath+= b.getLocation()+":";
			// TODO this fails when running from runnable jar :-(
			// we hack around it for now by finding where Concierge stores its jars
			// won't work on any OSGi framework though :'-(
			// TODO best way would probably be connecting to the repo and fetching api from there?
			if(b.getDataFile("")!=null)
				classpath+=b.getDataFile("").getParentFile().getAbsolutePath()+"/bundle0:";
		}
		compiler.run(null, null, null, "-classpath", classpath, sourceFile.getPath());
		
		// load class
		URLClassLoader classLoader = URLClassLoader.newInstance(new URL[] {root.toURI().toURL()}, wiring.getClassLoader());
		Class c = classLoader.loadClass(fqn);
		Object instance = c.newInstance();
		return (T) instance;
	}
	
}
