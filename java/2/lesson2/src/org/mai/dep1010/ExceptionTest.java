package org.mai.dep1010;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class ExceptionTest {
    public static void main(String[] args) {
        int a = 0;
        int b = 10;


        try {
            int c = b / a;
        } catch (ArithmeticException ae) {
            ae.printStackTrace();
        }


        String hello = "hello";
        System.out.println("hello = " + hello);
    }

    public static void openFile() throws IOException {
        File f = new File("test.txt");
        FileWriter fw = new FileWriter(f);
    }
}
