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
		
		// learn
		byte[] learnRequest = Files.readAllBytes(FileSystems.getDefault().getPath("test", "learnRequest"));
		out.write(learnRequest);
		
		JsonElement learnResult = parser.parse(reader);
		System.out.println(learnResult.toString());
		
		// eval
		byte[] evalRequest = Files.readAllBytes(FileSystems.getDefault().getPath("test", "evalRequest"));
		out.write(evalRequest);
		
		JsonElement evalResult = parser.parse(reader);
		System.out.println(evalResult.toString());
		
	}
	
}
