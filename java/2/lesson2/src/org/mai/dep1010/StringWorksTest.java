package org.mai.dep1010;

public class StringWorksTest {
    public static void main(String[] args) {

        String description = "The class String includes methods for examining\n" +
                " * individual characters of the sequence, for comparing strings, for\n" +
                " * searching strings, for extracting substrings, and for creating a\n" +
                " * copy of a string with all characters translated to uppercase or to\n" +
                " * lowercase";

        System.out.println("Length: "+description.length()+", is empty: "+description.isEmpty());

        String[] words = description.split(" ");
        for(String word : words) {
            System.out.println(word);
        }

        String shortDescription =  description.substring(10);//обрезает строку начиная с указанного индекса
        System.out.println(shortDescription);
        shortDescription =  description.substring(description.indexOf("String"));
        System.out.println(shortDescription);

        String usdNumber = "$101.78";
        System.out.println(usdNumber.matches("\\$\\d*\\.\\d*"));

        String stringWithPrice = "Cost is 101.56 USD in Target";
        System.out.println(stringWithPrice.replaceAll("(\\d*\\.\\d*)( USD)", "\\$$1"));

        String[] numbers = new String[] {"one", "two", "three"};
        System.out.println(String.join(",", numbers));


        //плохо
        String result = "0";
        for (int i = 1; i < 10000; i++) {
            result += "," + i;
        }
        System.out.println(result);


        //хорошо
        StringBuilder sb = new StringBuilder("0");
        for (int i = 0; i < 10000; i++) {
            sb.append(",").append(i);
        }
        System.out.println(sb.toString());

    }
}
