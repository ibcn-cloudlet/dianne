package be.iminds.iot.dianne.jsonrpc;

import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.file.FileSystems;
import java.nio.file.Files;

import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;

public class JSONRPCTester {

	public static void main(String[] args) throws Exception {
		
		Socket s = new Socket("127.0.0.1", 9090);
		
		OutputStream out = s.getOutputStream();
		out.flush();
		JsonReader reader = new JsonReader(new InputStreamReader(s.getInputStream()));
		JsonParser parser = new JsonParser();
		
		byte[] bytes = Files.readAllBytes(FileSystems.getDefault().getPath("test", "request"));
		out.write(bytes);
		
		JsonElement result = parser.parse(reader);
		System.out.println(result.toString());
	}
	
}
