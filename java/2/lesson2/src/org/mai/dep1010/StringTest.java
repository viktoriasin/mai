package org.mai.dep1010;

/**
 * Created by VPerov on 14.09.2018.
 */
public class StringTest {

    public static void main(String[] args) {
        /*

        //strings are interned
        String a1 = "test";
        String a2 = "test";
        System.out.println(a1 == a2); // == compare reference of objects .equals - compare the content
        System.out.println(a1.equals(a2));

        //string created as object
        String a3 = new String("test").intern(); //intern allow you to allocate the same memory to objects with equals content

        System.out.println(a1 == a3);
        System.out.println(a1.equals(a3));


        /*
        String value1 = "70";
        String value2 = "70";
        String value3 = new Integer(70).toString();
        Result:

        value1 == value2 ---> true

        value1 == value3 ---> false

        value1.equals(value3) ---> true

        value1 == value3.intern() ---> true
         */

        String s1 = "Hello, World";
        String s2 = "Hello, World";
        System.out.println("Comparing s1 and s2");
        System.out.println(s1 == s2);
        System.out.println(s1.equals(s2));

        System.out.println("Comparing s1 and s3");
        String s3 = new String(s1);
        System.out.println(s1 == s3);
        System.out.println(s1.equals(s3));

        System.out.println("Interning s1 and s3");
        System.out.println(s1 == s3.intern());

        System.out.println("Comparing concatenated strings");
        String s4 = s1+"!";
        System.out.println(s4 == s1+"!");
        System.out.println(s4 == (s1+"!").intern());
        System.out.println(s4.intern() == (s1+"!").intern());
        System.out.println(s4.equals(s1+"!"));


    }

}
