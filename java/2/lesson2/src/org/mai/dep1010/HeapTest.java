package org.mai.dep1010;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by VPerov on 14.09.2018.
 */
public class HeapTest {

    public static void main(String[] args) {

        List<String> millisList = new LinkedList<>();

        while(true){
            String millis = ""+System.currentTimeMillis();
            System.out.println("millis = " + millis);
            millisList.add(millis);
        }

    }
}
