package org.mai.dep1010.cer;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.math.BigDecimal;
import java.util.Currency;

/**
 * Created by VPerov on 17.09.2018.
 */
public class CurrencyExchangeRate {

    private BigDecimal rate;
    private Currency from;
    private Currency to;


    public CurrencyExchangeRate(BigDecimal rate, Currency from, Currency to) {
        this.rate = rate;
        this.from = from;
        this.to = to;
    }

    public Money convert(Money m) throws IncorrectExchangeRateException {
        if (m.getCurrency().getCurrencyCode().equals(this.to.getCurrencyCode())) {
            throw new IncorrectExchangeRateException("Exchange rate must be different!");
        } else {
            return new Money(this.to, m.getAmount().multiply(rate));
        }

    }
}
