package be.iminds.iot.dianne.dataset;

import static java.nio.file.StandardWatchEventKinds.ENTRY_CREATE;
import static java.nio.file.StandardWatchEventKinds.ENTRY_DELETE;
import static java.nio.file.StandardWatchEventKinds.ENTRY_MODIFY;
import static java.nio.file.StandardWatchEventKinds.OVERFLOW;

import java.io.Closeable;
import java.io.File;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.nio.file.WatchEvent;
import java.nio.file.WatchEvent.Kind;
import java.nio.file.WatchKey;
import java.nio.file.WatchService;
import java.util.function.Consumer;

public class DirectoryWatcher implements Closeable {

	private volatile boolean watching = true;
	
	public DirectoryWatcher(File dir, 
			Consumer<Path> created, 
			Consumer<Path> modified,
			Consumer<Path> deleted) {
		if(!dir.isDirectory())
			return;
		
		Thread t = new Thread(()->{
		
			Path path = dir.toPath();
			// We obtain the file system of the Path
			FileSystem fs = path.getFileSystem();
			
			// We create the new WatchService using the new try() block
			try(WatchService service = fs.newWatchService()) {
				
				// We register the path to the service
				// We watch for creation events
				path.register(service, ENTRY_CREATE,
                        ENTRY_DELETE,
                        ENTRY_MODIFY);
				
				// Start the infinite polling loop
				WatchKey key = null;
				while(watching) {
					key = service.take();
					
					// Dequeueing events
					Kind<?> kind = null;
					for(WatchEvent<?> watchEvent : key.pollEvents()) {
						// Get the type of the event
						kind = watchEvent.kind();
						if(kind == OVERFLOW){
							continue;
						} else if(kind ==  ENTRY_CREATE){
							Path newPath =  path.resolve(((WatchEvent<Path>) watchEvent).context());
							if(created!=null)
								created.accept(newPath);
						} else if(kind == ENTRY_MODIFY){
							Path modifiedPath = path.resolve(((WatchEvent<Path>) watchEvent).context());
							if(modified != null)
								modified.accept(modifiedPath);
						} else if(kind == ENTRY_DELETE){
							Path deletedPath = path.resolve(((WatchEvent<Path>) watchEvent).context());
							if(deleted!=null)
								deleted.accept(deletedPath);
						}		
					}
					
					if(!key.reset()) {
						break; //loop
					}
				}
				
			} catch(Exception ex) {
				ex.printStackTrace();
			} 
		});
		t.start();
	}
	
	
	@Override
	public void close() {
		watching = false;
	}
	
}
