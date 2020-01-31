package org.mai.dep1010;

import java.math.BigDecimal;

/**
 * Created by VPerov on 14.09.2018.
 */
public class PrimitiveNumbers {

    public static void main(String[] args) {
        doubleTest();
        floatTest();
        bigDecimalTest();
    }

    private static void doubleTest() {
        double d = 0.1;
        double result = 0.0;
        for (int i = 0; i < 10; i++) {
            result += d;
        }

        System.out.println("result = " + result);
    }

    private static void floatTest() {
        float d = 0.1f;
        float result = 0.0f;
        for (int i = 0; i < 10; i++) {
            result += d;
        }

        System.out.println("result = " + result);
    }

    private static void bigDecimalTest() {
        BigDecimal d = new BigDecimal("0.1");
        BigDecimal result = new BigDecimal(0).setScale(10, BigDecimal.ROUND_UP);

        for(int i = 0; i < 10; i++) {
            result = result.add(d);
        }

        System.out.println("result: "+result);
    }


}

















