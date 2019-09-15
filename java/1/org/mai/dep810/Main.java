package org.mai.dep810;

/*
Multiline
comment
example
*/

import java.util.Scanner;

public class Main {
    public static void main(String[] args){

        //Iterating through input argumets
        for(String arg : args) {
            System.out.println(arg);
        }

        String greating = System.getProperty("org.mai.dep810.greating");
        if(greating != null) {
            System.out.println(greating);
        }

        //reading from console input
        Scanner s = new Scanner(System.in);
        System.out.println("Enter something:");
        while(s.hasNext()) {
            String line = s.nextLine();
            System.out.println(line);
            if(line.equals("exit")) {
                break;
            }
        }
    }
}
