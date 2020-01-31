package org.mai.dep1010;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by VPerov on 18.01.2019.
 */
public class GenericTest {
    public static void main(String[] args) {
        System.out.println("Start");

        List a = new ArrayList();
        List<Integer> b = new ArrayList<>();

        a.add(new Object());
        //b.add(10);

        b = a;

        System.out.println(b);



        System.out.println("Finish");
    }
}
