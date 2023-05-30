/***********************************************
**                MSc ANALYTICS 
**     DATA ENGINEERING PLATFORMS (MSCA 31012)
** File:   Sakila Snowflake DDL - Assignment 5
** Desc:   ETL/DML for the Sakila Snowflake Dimensional model
** Auth:   Shreenidhi Bharadwaj, Ashish Pujari, Audrey Salerno
** Date:   04/08/2018, Last updated 02/09/2021
************************************************/

---------------------------------------
USE SAKILA_SNOWFLAKE;

-- -----------------------------------------------------
-- Populate Time dimension
-- -----------------------------------------------------

INSERT INTO numbers_small VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9);

INSERT INTO numbers
SELECT 
    thousands.number * 1000 + hundreds.number * 100 + tens.number * 10 + ones.number
FROM
    numbers_small thousands,
    numbers_small hundreds,
    numbers_small tens,
    numbers_small ones
LIMIT 1000000;

INSERT INTO dim_date (date_key, date)
SELECT 
    number,
    DATE_ADD('2005-01-01',
        INTERVAL number DAY)
FROM
    numbers
WHERE
    DATE_ADD('2005-01-01',
        INTERVAL number DAY) BETWEEN '2005-01-01' AND '2017-01-01'
ORDER BY number;

SET SQL_SAFE_UPDATES = 0;
UPDATE dim_date 
SET 
    timestamp = UNIX_TIMESTAMP(date),
    day_of_week = DATE_FORMAT(date, '%W'),
    weekend = IF(DATE_FORMAT(date, '%W') IN ('Saturday' , 'Sunday'),
        'Weekend',
        'Weekday'),
    month = DATE_FORMAT(date, '%M'),
    year = DATE_FORMAT(date, '%Y'),
    month_day = DATE_FORMAT(date, '%d');

UPDATE dim_date 
SET 
    week_starting_monday = DATE_FORMAT(date, '%v');

-- -----------------------------------------------------
-- Copy Data from sakila database 
-- -----------------------------------------------------
# dim_actor table 
# insert data into the dim_actor table from sakila models actor table 
INSERT INTO sakila_snowflake.dim_actor (    
	actor_id,
    actor_first_name,
    actor_last_name,
    actor_last_update)
(SELECT 
    actor_id,
    first_name,
    last_name,
    last_update
FROM
    sakila.actor);
    
# dim_staff table
# insert data into the dim_staff table from sakila models staff table 
INSERT INTO sakila_snowflake.dim_staff (    
	staff_id,
    staff_first_name,
    staff_last_name,
    staff_store_id,
    staff_last_update)
(SELECT 
    staff_id,
    first_name,
    last_name,
    store_id,
    last_update
FROM
    sakila.staff);


# dim_location_country
# insert data into the dim_location_country table from sakila models country table 
INSERT INTO sakila_snowflake.dim_location_country (    
	location_country_id,
    location_country_name,
    location_country_last_update)
(SELECT 
    country_id,
    country,
    last_update
FROM
    sakila.country);

# dim_location_city
# insert data into the dim_location_city table from sakila model city table 
INSERT INTO sakila_snowflake.dim_location_city(
	location_country_key,
	location_city_id,
    location_city_name,
    location_city_last_update) 
(SELECT
    location_country_key,
    city_id,
	city,
    s_city.last_update
FROM
    sakila.country as s_country, 
    sakila.city as s_city,
    sakila_snowflake.dim_location_country as dim_country
WHERE
	s_city.country_id = s_country.country_id AND
    s_city.country_id = dim_country.location_country_id);


# dim_location_address
# insert data into the dim_location_address table from sakila model city table 
INSERT INTO sakila_snowflake.dim_location_address(
	location_city_key,
	location_address_id,
    location_address,
    location_address_postal_code,
    location_address_last_update)
(SELECT
	dim_city.location_city_key,
	address_id,
    address,
    postal_code,
    last_update
FROM
    sakila.address as s_address,
    sakila_snowflake.dim_location_city as dim_city
WHERE
    s_address.city_id = dim_city.location_city_id);


# dim_store table
# insert data into the dim_store table from sakila models staff table 
INSERT INTO sakila_snowflake.dim_store (    
	location_address_key,
    store_last_update,
    store_id,
    store_manager_staff_id,
    store_manager_first_name,
    store_manager_last_name)
(SELECT 
    s_addr.location_address_key,
    s_store.last_update,
    s_store.store_id,
    manager_staff_id,
    first_name,
    last_name
FROM
    sakila.store as s_store,
    dim_location_address as s_addr,
    sakila.staff as s_staff
WHERE 
	s_store.address_id = s_addr.location_address_id  AND
    s_staff.staff_id = s_store.manager_staff_id);


# dim_customer
INSERT INTO sakila_snowflake.dim_customer (    
	location_address_key,
    customer_last_update,
    customer_id,
    customer_first_name,
    customer_last_name,
    customer_email,
    customer_active,
    customer_created)
(SELECT
    location_address_key,
    last_update,
    customer_id,
    first_name,
    last_name,
    email,
    active,
    create_date
FROM
    sakila.customer as s_cust,
    dim_location_address as s_addr
WHERE 
	s_cust.address_id = s_addr.location_address_id);

    

# dim_film
# insert data into the dim_store table from sakila models staff table 
INSERT INTO sakila_snowflake.dim_film (    
	film_id,
    film_last_update,
    film_title,
    film_description,
    film_release_year,
    film_language,
    film_rental_duration,
    film_rental_rate,
    film_duration,
    film_replacement_cost,
    film_rating_code,
    film_rating_text,
    film_has_trailers,
    film_has_commentaries,
    film_has_deleted_scenes,
    film_has_behind_the_scenes,
    film_in_category_action,
    film_in_category_animation,
    film_in_category_children,
    film_in_category_classics,
    film_in_category_comedy,
    film_in_category_documentary,
    film_in_category_drama,
    film_in_category_family,
    film_in_category_foreign,
    film_in_category_games,
    film_in_category_horror,
    film_in_category_music,
    film_in_category_new,
    film_in_category_scifi,
    film_in_category_sports,
    film_in_category_travel )
(
select f.film_id,
    f.last_update,
    f.title,
    f.description,
    f.release_year,
    l.name,
    f.rental_duration AS film_rental_duration,
    f.rental_rate AS film_rental_rate,
    f.length AS film_duration,
    f.replacement_cost AS film_replacement_cost,
    f.rating AS film_rating_code,
    f.special_features AS film_rating_text,   
	case when  f.special_features like '%Commentaries%' then 1 else 0 end film_has_Commentaries,
    case when  f.special_features like '%Trailers%' then 1 else 0 end film_has_trailers,
    case when  f.special_features like '%Deleted Scenes%' then 1 else 0 end film_has_deleted_scenes,
    case when  f.special_features like '%Behind the Scenes%' then 1 else 0 end film_has_behind_the_scenes,
	case when  c.name ='Action' THEN 1 ELSE 0 END AS film_in_category_action,
	case when  c.name ='Animation' THEN 1 ELSE 0 END AS film_in_category_animation,
	case when  c.name ='Children' THEN 1 ELSE 0 END AS film_in_category_children,
	case when  c.name ='Classics' THEN 1 ELSE 0 END AS  film_in_category_classics,
	case when  c.name ='Comedy' THEN 1 ELSE 0 END AS  film_in_category_comedy,
	case when  c.name ='Documentary' THEN 1 ELSE 0 END AS  film_in_category_documentary,
	case when  c.name ='Drama' THEN 1 ELSE 0 END AS film_in_category_drama,
	case when  c.name ='Family' THEN 1 ELSE 0 END AS film_in_category_family,
	case when  c.name ='Foreign' THEN 1 ELSE 0 END AS film_in_category_foreign,
	case when  c.name ='Games' THEN 1 ELSE 0 END AS film_in_category_games,
	case when  c.name ='Horror' THEN 1 ELSE 0 END AS film_in_category_horror,
	case when  c.name ='Music' THEN 1 ELSE 0 END AS  film_in_category_music,
	case when  c.name ='New' THEN 1 ELSE 0 END AS film_in_category_new,
	case when  c.name ='Sci-Fi' THEN 1 ELSE 0 END AS film_in_category_scifi,
	case when  c.name ='Sports' THEN 1 ELSE 0 END AS film_in_category_sports,
	case when  c.name ='Travel' THEN 1 ELSE 0 END AS film_in_category_travel 
  from sakila.category c , sakila.film f, sakila.film_category fc, sakila.language l
  where  f.film_id = fc.film_id  and
  f.language_id = l.language_id and
  c.category_id = fc.category_id);


# dim_actor_bridge
INSERT INTO sakila_snowflake.dim_film_actor_bridge (    
	film_key,
    actor_key
)
(SELECT
     film_key,
     actor_key
FROM
    sakila.film_actor s_fa,
    dim_actor d_a, 
    dim_film d_f
WHERE 
	s_fa.actor_id = d_a.actor_id AND
	s_fa.film_id = d_f.film_id);


# The below query might take over 30 seconds to complete and you might get an "Error Code: 2013. 
# Lost connection to MySQL server during query" error
# Please follow the instructions below:
#   - In the application menu, select Edit > Preferences > SQL Editor.
#   - Look for the MySQL Session section and increase the DBMS connection read time out value.
#   - Save the settings, quit MySQL Workbench and reopen the connection.

-- -----------------------------------------------------
-- Write Fact table fact_rental DML script here
-- -----------------------------------------------------

INSERT INTO sakila_snowflake.fact_rental (
  rental_last_update, customer_key, 
  staff_key, film_key, store_key, rental_date_key, 
  return_date_key, count_rentals, 
  count_returns, rental_duration, 
  dollar_amount
) 
SELECT 
  r.LAST_UPDATE, 
  c.customer_key, 
  s.staff_key, 
  f.film_key, 
  st.store_key, 
  d.date_key, 
  (
    CASE WHEN r.return_date IS NOT NULL THEN (
      SELECT 
        d1.date_key 
      FROM 
        sakila_snowflake.dim_date AS d1 
      WHERE 
        d1.date = DATE(r.return_date)
    ) END
  ), 
  COUNT(r.rental_id), 
  (
    CASE WHEN r.return_date IS NOT NULL THEN COUNT(r.rental_id) ELSE 0 END
  ), 
  f.film_rental_duration, 
  p.amount 
FROM 
  sakila.rental AS r, 
  sakila_snowflake.dim_customer AS c, 
  sakila_snowflake.dim_staff AS s, 
  sakila_snowflake.dim_film AS f, 
  sakila_snowflake.dim_store AS st, 
  sakila.inventory AS i, 
  sakila_snowflake.dim_date AS d, 
  sakila.payment AS p 
WHERE 
  i.inventory_id = r.inventory_id 
  AND c.customer_id = r.customer_id 
  AND s.staff_id = r.staff_id 
  AND f.film_id = i.film_id 
  AND st.store_id = i.store_id 
  AND d.date = DATE(r.rental_date) 
  AND p.rental_id = r.rental_id 
GROUP BY 
  r.rental_id;

