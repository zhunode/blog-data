import java.io.*;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataInputStream;


/**
 * 读取hadoop hdfs文件流，保存到本地
 */
public class Chapter3 {
    public static void main(String[] args) {
        try {
            /*
            * java.lang.IllegalArgumentException: Wrong FS: hdfs://master:9000/user/test/test.txt, expected: file:///
            * 参考：https://stackoverflow.com/questions/32078441/wrong-fs-expected-file-when-trying-to-read-file-from-hdfs-in-java
            * 按照文档所说的进行书写代码 copy.
             * */
            Configuration conf = new Configuration();
            FSDataInputStream fsDataInputStream;
            FileSystem fs = FileSystem.get(new URI("hdfs://172.16.0.130:9000"), conf);
            Path filePath = new Path("hdfs://172.16.0.130:9000/user/test/test.txt");
            if(fs.exists(filePath)){
                FSDataInputStream is = fs.open(filePath);
                byte[] buff = new byte[1024];
                int length = 0;
                OutputStream os=new FileOutputStream(new File("D:/a.txt"));
                while ((length=is.read(buff))!=-1){
                    System.out.println(new String(buff,0,length));
                    os.write(buff,0,length);
                    os.flush();
                }
                System.out.println("文件存在");
                System.out.println(fs.getClass().getName());

                // 关闭流
                os.close();
                is.close();
                fs.close();
            }else{
                System.out.println("文件不存在");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }


//            FSDataInputStream is = fs.open(new Path("hdfs://172.16.0.130:9000/home/hadoop/test.txt"));
//            OutputStream os=new FileOutputStream(new File("D:/a.txt"));
//            byte[] buff= new byte[1024];
//            int length = 0;
//            while ((length=is.read(buff))!=-1){
//                System.out.println(new String(buff,0,length));
//                os.write(buff,0,length);
//                os.flush();
//            }
//            System.out.println(fs.getClass().getName());
//            //这个是根据你传的变量来决定这个对象的实现类是哪个
//        }catch(Exception e){
//            e.printStackTrace();
//        }
    }
}
