package org.mai.dep810.basket;

import java.util.Map;

public interface Basket {
    /* добавить продукт в корзину с заданным количеством */
    void addProduct(String product, int quantity);

    /* удалить продукт из корзины */
    void removeProduct(String product);

    /* изменить количество продукта */
    void updateProductQuantity(String product, int quantity);

    /* получить список продуктов */
    Map<String, Integer> getProducts();
}
