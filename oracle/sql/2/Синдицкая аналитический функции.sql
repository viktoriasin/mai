
-- 1) Каждый месяц компания выдает премию в размере 5% от суммы продаж менеджеру(одному или нескольким, в случае равных результатов с другими
-- менеджерами?), который
-- за предыдущие 3 месяца продал товаров на самую большую сумму
-- Выведите месяц, manager_id, manager_first_name, manager_last_name,
-- премию за период с января по декабрь 2014 года

select month_ - 2 month,  manager_id, manager_first_name,  manager_last_name, avg_sum * 0.05 salary
from (
      select trunc(MONTHS_BETWEEN(sale_date, '01.10.2013')) month_
         , manager_id
         , manager_first_name
         , manager_last_name
         , sale_amount
         , avg_sum
         , dense_rank() over (partition by trunc(MONTHS_BETWEEN(sale_date, '01.10.2013')) order by avg_sum desc) k  --row_number если нужен только 1 менеджер
         from (
               select sale_date
        , sale_amount
        , manager_id
        , manager_first_name
        , manager_last_name
        , sum(sale_amount)
         over (partition by manager_id  ORDER BY sale_date range between INTERVAL '3' MONTH PRECEDING AND current row ) avg_sum
         from OLTP_TEST.v_fact_sale
         where sale_date between '01.10.2013' and '31.12.2014'
         )
         where sale_date between '01.01.2014' and '31.12.2014'
         ) where k = 1
group by  month_ - 2, manager_id, manager_first_name,  manager_last_name, avg_sum
order by month
;

-- 2) Компания хочет оптимизировать количество офисов, проанализировав относительные объемы продаж по офисам в течение периода с 2013-2014 гг.
-- Выведите год, office_id, city_name, country, относительный объем продаж за текущий год
-- Офисы, которые демонстрируют наименьший относительной объем в течение двух лет скорее всего будут закрыты.

SELECT extract(YEAR from sale_date) year
     , office_id
     , city_name
     , country
     , RATIO_TO_REPORT(sum(sale_amount)) OVER () AS ratio
FROM OLTP_TEST.v_fact_sale
where extract(YEAR from sale_date) between 2013 and 2014 and office_id is not null
group by extract(YEAR from sale_date)
       , office_id
       , city_name
       , country
order by year, ratio desc;

-- 3)  Для планирования закупок, компанию оценивает динамику роста продаж по товарам.
-- Динамика оценивается как отношение объема продаж в текущем месяце к предыдущему.
-- Выведите товары, которые демонстрировали наиболее высокие темпы роста продаж в течение первого полугодия 2014 года.

select month_, product_id, product_name
from (
         select month_,
                product_id,
                product_name,
                row_number() over(partition by month_ order by ratio desc) top
         from (
                  select trunc(sale_date, 'mm') month_
                       , product_id
                       , product_name
                       , nvl(sum(sale_amount) / lag(sum(sale_amount)) over (partition by product_id order by trunc(sale_date, 'mm')), 0) ratio
                  from OLTP_TEST.v_fact_sale
                  where sale_date between '01.12.2013' and '30.06.2014'
                  group by trunc(sale_date, 'mm')
                         , product_id
                         , product_name
              )
         where month_ between '01.01.2014' and '30.06.2014'
     ) where top = 1;

-- 4) Напишите запрос, который выводит отчет о прибыли компании за 2014 год: помесячно и поквартально.
-- Отчет включает сумму прибыли за период и накопительную сумму прибыли с начала года по текущий период.

select
    month_
     , quarter_
     , sale_amount sum_month
     , sum(sale_amount) over (partition by quarter_) sum_quart
     , sum(sale_amount) over (order by month_ rows between unbounded preceding and current row) c_sum
     , sum(sale_amount) over (order by quarter_ rows between unbounded preceding and current row) q_sum
from (
         select
             to_char(sale_date, 'MM') month_
              , to_char(sale_date, 'Q') quarter_
              , sum(SALE_AMOUNT) sale_amount
         from OLTP_TEST.v_fact_sale
         where sale_date between '01.01.2014' and '31.12.2014'
         group by to_char(sale_date, 'MM') , to_char(sale_date, 'Q'))
order by  month_
;

--  5. Найдите вклад в общую прибыль за 2014 год 10% наиболее дорогих товаров и 10% наиболее дешевых товаров.
--     Выведите product_id, product_name, total_sale_amount, percent

select product_id
     , product_name
     , total_sale_amount
     , round(ratio*100, 4) persent
from (
         select product_id
              , product_name
              , sum(sale_amount) total_sale_amount
              , cume_dist() over ( order by sum(sale_amount)) dist
              , RATIO_TO_REPORT(sum(sale_amount)) OVER () AS ratio
         from OLTP_TEST.v_fact_sale
         where sale_date between '01.01.2014' and '31.12.2014'
         group by product_id, PRODUCT_NAME
     )
where dist >= 0.9 or dist <= 0.1 ;

--     6. Компания хочет премировать трех наиболее продуктивных (по объему продаж, конечно) менеджеров в каждой стране в 2014 году.
-- Выведите country, <список manager_last_name manager_first_name, разделенный запятыми> которым будет выплачена премия

select country, listagg(manager_last_name || ' ' || manager_first_name, ', ') within group (order by manager_last_name) from (
                                                                                                                                 select country
                                                                                                                                      , manager_id
                                                                                                                                      , manager_first_name
                                                                                                                                      , manager_last_name
                                                                                                                                      , sum(sale_amount)
                                                                                                                                      , row_number() over (partition by country order by sum(sale_amount) desc) rnk
                                                                                                                                 from OLTP_TEST.v_fact_sale
                                                                                                                                 where extract(year from sale_date) = 2014
                                                                                                                                 group by country, manager_id, manager_first_name, manager_last_name
                                                                                                                             )
where rnk <= 3
group by country;

--  7. Выведите самый дешевый и самый дорогой товар, проданный за каждый месяц в течение 2014 года.
-- cheapest_product_id, cheapest_product_name, expensive_product_id, expensive_product_name, month, cheapest_price, expensive_price



select distinct to_char(sale_date, 'MM') month_,
                last_value(product_id) over(partition by to_char(sale_date, 'MM') order by sale_price rows between unbounded preceding  and unbounded following ) expensive_product_id,
                last_value(product_name) over(partition by to_char(sale_date, 'MM') order by sale_price rows between unbounded preceding  and unbounded following ) expensive_product_name,
                first_value(product_id) over(partition by to_char(sale_date, 'MM') order by sale_price rows between unbounded preceding  and unbounded following ) cheapest_product_id,
                first_value(product_name) over(partition by to_char(sale_date, 'MM') order by sale_price rows between unbounded preceding  and unbounded following ) cheapest_product_name,
        max(sale_price) over(partition by to_char(sale_date, 'MM')) expensive_price,
       min(sale_price) over(partition by to_char(sale_date, 'MM'))  cheapest_price
from v_fact_sale
where sale_date between '01.01.2014' and '31.12.2014'

--8

select month_,
       round(sum(sale_amount),2) sale_amount,
       round(sum(salary),2) salary,
       round(sum(profit),2) profit


from (
         select distinct extract(MONTH from sale_date)                                                        month_
              , sum(sale_amount) over (partition by extract(MONTH from sale_date) , manager_id)               sale_amount
              , 30000 + 0.05 * (sum(sale_amount) over (partition by extract(MONTH from sale_date) , manager_id)) salary
              , sum(sale_amount) over (partition by extract(MONTH from sale_date) , manager_id) - 30000 -
                0.05 * (sum(sale_amount) over (partition by extract(MONTH from sale_date) , manager_id)) -
                 (sum( sale_price/1.1 * sale_qty) over (partition by extract(MONTH from sale_date) , manager_id)) profit
         from OLTP_TEST.v_fact_sale
     )
group by month_
order by month_

