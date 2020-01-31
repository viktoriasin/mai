package org.mai.dep1010.cer;

import java.math.BigDecimal;
import java.util.Currency;
import java.util.List;

/**
 * Created by Asus on 9/17/2018.
 */
public class Main {
    public static void main(String[] args) {
        Currency usd = Currency.getInstance("USD");
        Currency gbp = Currency.getInstance("GBP");

        Money usdMoney = new Money(usd, new BigDecimal(100));
        Money tenDollars = new Money(usd, new BigDecimal(10));
        Money tenPound = new Money(gbp, new BigDecimal(10));
        CurrencyExchangeRate poundToUsd = new CurrencyExchangeRate(new BigDecimal(1.5), gbp, usd);

        //should set usdMoney 110 with scale 2
        usdMoney = usdMoney.add(tenDollars);
        System.out.println(usdMoney.getAmount().equals(new BigDecimal(110).setScale(2)));

        //should throw DifferentCurrenciesException
        try {
            usdMoney = usdMoney.subtract(tenPound);
        } catch(DifferentCurrenciesException ex) {
            System.out.println("DifferentCurrenciesException thrown");
        }

        //System.out.println(poundToUsd.convert(tenPound).getAmount());

        //should set usdMoney 95 with scale 2
        usdMoney = usdMoney.subtract(poundToUsd.convert(tenPound));
        System.out.println(usdMoney.getAmount().equals(new BigDecimal(95).setScale(2)));



        //////////////////////////////////////////////////////////////////////////
        //20 / 7 15//7
        Money eqTest = new Money(usd, new BigDecimal((13)));
        int i = 9;

        List<Money> mn = eqTest.parts(i);

        for (int j = 0; j < i; j++) {
            System.out.println(mn.get(j).getAmount());
        }

        BigDecimal eqRes = new BigDecimal((0));
        for (int j = 0; j < i; j++) {
            eqRes = eqRes.add(mn.get(j).getAmount());
        }
        System.out.println("Result of eqTest is " + eqRes.toString());
        //////////////////////////////////////////////////////////////////////////



        Money m07 = tenDollars.multiply(new BigDecimal(0.07));

        System.out.println("m07 = " + m07.getAmount());


        //Money res = new Money(usd, eqTest.devide(new BigDecimal(i)).getAmount());
        //Money res = new Money(usd, eqTest.parts_divide(new BigDecimal(i)).getAmount());
        //Money z = res.add(res).add(res);
        //System.out.println(z.getAmount());
        //System.out.println(eqTest.devide(new BigDecimal(i)).getAmount());
        //System.out.println(new Money(usd, res).multiply(new BigDecimal(i)).getAmount());

        //делим на части, смотрим сколько осталось и остатки раздлеляем рандомно



    }
}
