package org.mai.dep1010.cer;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.math.BigDecimal;
import java.util.*;

public class Money {
    private Currency currency;
    private BigDecimal amount;

    public Money(Currency currency, BigDecimal amount) {
        this.currency = currency;
        this.amount = amount.setScale(this.currency.getDefaultFractionDigits());
    }

    public Currency getCurrency() {
        return currency;
    }

    public BigDecimal getAmount() {
        return amount;
    }

    public Money add(Money m) throws DifferentCurrenciesException {
        if (!this.currency.equals(m.getCurrency())) {
            throw new DifferentCurrenciesException("The currencies of money must be the same!");
        } else {
            return new Money(this.currency, this.amount.add(m.getAmount()));
        }
    }

    public Money subtract(Money m) throws DifferentCurrenciesException {
        if (!this.currency.equals(m.getCurrency())) {
            throw new DifferentCurrenciesException("The currencies of money must be the same!");
        } else {
            return new Money(this.currency, this.amount.subtract(m.getAmount()));
        }
    }

    public Money multiply(BigDecimal ratio) {
        return new Money(this.currency, this.amount.multiply(ratio).setScale(this.currency.getDefaultFractionDigits(), BigDecimal.ROUND_HALF_UP));
    }

    public Money devide(BigDecimal ratio) {
        return new Money(this.currency, this.amount.divide(ratio, this.amount.scale(), BigDecimal.ROUND_HALF_UP));
    }

    public List<Money> parts(int n) {
        List<Money> lst_m = new ArrayList<>();
        BigDecimal m = new BigDecimal((int) n);

//        for (int i = 0; i < n - 1 ; i++) {
//            lst_m.add(new Money(this.currency, this.amount.divide(new BigDecimal((int) n), BigDecimal.ROUND_HALF_DOWN)));
//        }
//
//        Money last = new Money(this.currency, this.amount.subtract(this.amount.divide(new BigDecimal((int) n), BigDecimal.ROUND_HALF_DOWN).multiply(new BigDecimal((int) (n -1)))));
//        lst_m.add(last);

        BigDecimal average = this.amount.divide(new BigDecimal((int) n), BigDecimal.ROUND_HALF_DOWN);
        BigDecimal extra = average.multiply(m).subtract(amount).setScale(2,BigDecimal.ROUND_HALF_DOWN);
        System.out.println(extra.toString());
        for (int i = 0; i < n ; i++) {
            if ((extra.compareTo(BigDecimal.ZERO) != 0)) {
                if (extra.compareTo(BigDecimal.ZERO) < 0) {
                    lst_m.add(new Money(this.currency, this.amount.divide(new BigDecimal((int) n), BigDecimal.ROUND_HALF_DOWN).add(new BigDecimal(0.01)).setScale(2, BigDecimal.ROUND_HALF_DOWN)));
                    extra = extra.add(new BigDecimal(0.01)).setScale(2, BigDecimal.ROUND_HALF_DOWN);
                    System.out.println(extra.toString());
                } else {
                    lst_m.add(new Money(this.currency, this.amount.divide(new BigDecimal((int) n), BigDecimal.ROUND_HALF_DOWN).subtract(new BigDecimal(0.01)).setScale(2, BigDecimal.ROUND_HALF_DOWN)));
                    extra = extra.subtract(new BigDecimal(0.01)).setScale(2, BigDecimal.ROUND_HALF_DOWN);
                    System.out.println(extra.toString());
                }
            } else {
                lst_m.add(new Money(this.currency, this.amount.divide(new BigDecimal((int) n), BigDecimal.ROUND_HALF_DOWN)));
            }
        }
        return lst_m;
    }
}



