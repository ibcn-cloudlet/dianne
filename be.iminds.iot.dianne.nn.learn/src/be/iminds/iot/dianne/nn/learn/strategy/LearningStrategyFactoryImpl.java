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
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn.strategy;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

import org.osgi.framework.Bundle;
import org.osgi.framework.BundleContext;
import org.osgi.framework.wiring.BundleWiring;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategyFactory;

@Component
public class LearningStrategyFactoryImpl implements LearningStrategyFactory {

	private BundleContext context;
	private BundleWiring wiring;
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
		this.wiring = context.getBundle().adapt(BundleWiring.class);
	}
	
	public LearningStrategy createLearningStrategy(String strategy){
		if(strategy.contains("class")){
			try {
				return createLearningStrategyFromCode(strategy);
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
				List<URL> urls = wiring.findEntries("/", strategy+".class", BundleWiring.FINDENTRIES_RECURSE);
				
				if(urls.size() == 0){
					System.err.println("LearningStrategy "+strategy+" not found");
					return null;
				}
				
				String u = urls.get(0).toString().substring(9);
				u = u.substring(u.indexOf("/")+1, u.length()-6);
				u = u.replaceAll("/", ".");
		
				c = this.getClass().getClassLoader().loadClass(u);
			}
			
			return (LearningStrategy) c.newInstance();
		} catch (Throwable e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	public LearningStrategy createLearningStrategyFromCode(String source) throws Exception {
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
		// put all bundles on the compiler classpath
		String classpath = "";
		for(Bundle b : context.getBundles()){
			classpath+= b.getLocation()+":";
		}
		compiler.run(null, null, null, "-classpath", classpath, sourceFile.getPath());
		
		// load class
		URLClassLoader classLoader = URLClassLoader.newInstance(new URL[] {root.toURI().toURL()}, wiring.getClassLoader());
		Class c = classLoader.loadClass(fqn);
		Object instance = c.newInstance();
		return (LearningStrategy) instance;
	}
	
}
