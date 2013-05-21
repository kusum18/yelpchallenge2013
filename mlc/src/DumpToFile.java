
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
 
public  class DumpToFile {
	public static void dumpResults(String results, String fileName ) 
	{
		try {
 				File file = new File(fileName);

				// if file doesnt exists, then create it
				if (!file.exists()) {
					file.createNewFile();
				}

				FileWriter fw = new FileWriter(file.getAbsoluteFile(), true);
				BufferedWriter bw = new BufferedWriter(fw);
				bw.write(results);
				bw.close();

				System.out.println("Done");

		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}