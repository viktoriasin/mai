package org.mai.dep1010;

public class StackTest {

    public static void main(String[] args) {
        System.out.println("Start");

        Object o = new Object();
        StackTest test = new StackTest();

        test.a(o);

        System.out.println("Finish");
    }

    void a(Object p) {
        String objectToString = p.toString();
        Object p2 = new Object();
        b(objectToString);
    }

    void b(String s) {
        int stringLength = s.length();
        c(stringLength);
    }

    void c(int i) {
        System.out.println("i = " + i);
    }

}
