package org.mai.dep1010.quantity;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.math.BigDecimal;

public class Quantity {
    private BigDecimal value;
    private UnitOfMeasure measure;

    public Quantity(BigDecimal value, UnitOfMeasure measure) {
        this.value = value;
        this.measure = measure;
    }

    public BigDecimal getValue() {
        return value;
    }

    public UnitOfMeasure getMeasure() {
        return measure;
    }

    public Quantity add(Quantity other) {
        return new Quantity(this.value.add(other.getValue()), this.measure);
    }

    public Quantity subtract(Quantity other) {
        return new Quantity(this.value.subtract(other.getValue()), this.measure);
    }

    public Quantity multiply(BigDecimal ratio) {
        return  new Quantity(this.value.multiply(ratio), this.measure);
    }

    public Quantity devide(BigDecimal ratio) {
        return  new Quantity(this.value.divide(ratio, this.value.scale()), this.measure);

    }

    public String toString() {
        return value + " " + measure;
    }

}
