package org.mai.dep810.collections.basket;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class BasketOnline implements Basket {

    private Map<String, Integer> basket = new HashMap<>();

    /* добавить продукт в корзину с заданным количеством */
    @Override
    public void addProduct(String product, int quantity) {
        if (product == null || product.trim().length() == 0) {
            throw  new IllegalArgumentException("Название продукта не должно быть пустым");
        }
        if (quantity <= 0) {
            throw  new IllegalArgumentException("Количество продукта должно быть больше нуля");
        }
        if (basket.containsKey(product)) {
            basket.put(product,basket.get(product) + quantity);
        } else {
            basket.put(product, quantity);
        }
    };

    /* удалить продукт из корзины */
    @Override
    public void removeProduct(String product) {
        if (basket.remove(product) == null) {
            throw new IllegalArgumentException("Продукта '" + product + "' нет в корзине");
        };
    };

    /* изменить количество продукта */
    @Override
    public void updateProductQuantity(String product, int quantity) {
        if (!basket.containsKey(product)) {
            throw new IllegalArgumentException("Продукта '" + product + "' нет в корзине");
        }
        if (quantity <= 0) {
            throw  new IllegalArgumentException("Количество продукта должно быть больше нуля");
        }
        basket.put(product, quantity);

    };

    /* получить список продуктов */
    @Override
    public List<String> getProducts() {
        return new ArrayList<>(basket.keySet());

    }
}
