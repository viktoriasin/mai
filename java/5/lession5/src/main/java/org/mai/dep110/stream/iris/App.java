package org.mai.dep110.stream.iris;

import org.mai.dep110.stream.employees.Employee;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.mai.dep110.stream.iris.Iris.*;


public class App {
    public static void main(String[] args) throws IOException {
        App a = new App();
        a.test();
    }

    private static String SpeciesGet(Iris t) {
        return t.getSpecies();
    }

    private static Petal PatelClassify(Iris t) {
        return t.classifyByPatel();
    }

    public void test() throws IOException {

        List<Iris> irisList = Files
                .lines(Paths.get("iris.data"))
                .map(Iris::parse)
                .collect(Collectors.toList());  //load data from file iris.data
        IrisDataSetHelper helper = new IrisDataSetHelper(irisList);

        //get average sepal width
        Double avgSepalLength = helper.getAverage(Iris::getSepalWidth);
        System.out.println(avgSepalLength);

        //get average petal square - petal width multiplied on petal length
        Double avgPetalLength = helper.getAverage(p -> (p.getPetalLength()  * p.getPetalLength()));
        System.out.println(avgPetalLength);

        //get average petal square for flowers with sepal width > 4
        Double avgPetalSquare = helper.getAverageWithFilter(p -> p.getSepalWidth() > 4, p -> (p.getPetalLength()  * p.getPetalLength()));
        System.out.println(avgPetalSquare);

        //get flowers grouped by Petal size (Petal.SMALL, etc.)
        Map groupsByPetalSize = helper.groupBy(p -> PatelClassify((Iris) p));
        System.out.println(groupsByPetalSize);

        //get max sepal width for flowers grouped by species
        Map maxSepalWidthForGroupsBySpecies = helper.maxFromGroupedBy(Iris::getSpecies, Iris::getSepalWidth);
        System.out.println(maxSepalWidthForGroupsBySpecies);
    }


}

