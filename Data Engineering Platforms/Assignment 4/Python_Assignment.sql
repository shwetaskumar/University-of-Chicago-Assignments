#List all addresses with country_id = 44. Return the address_id, address line 1, district, city, country and postal code
SELECT a.address_id, address, district, city.city, c.country, a.postal_code FROM sakila.address AS a LEFT JOIN sakila.city AS city ON a.city_id = city.city_id LEFT JOIN sakila.country AS c ON city.country_id = c.country_id WHERE c.country_id = 44;

#Return the staff_id, first name, last name and the avaerage sales of every staff member. Alias the average sales as avg_sales.
SELECT p.staff_id, s.first_name, s.last_name, AVG(p.amount) AS avg_sales FROM sakila.payment AS p LEFT JOIN sakila.staff AS s ON p.staff_id = s.staff_id GROUP BY p.staff_id;

#List the actors with sales greater than $1000. Return their first name, last name, actor_id and sales amount aliased as sales. Order the results by the sales generated.
SELECT a.first_name, a.last_name, a.actor_id, SUM(p.amount) AS sales FROM sakila.payment AS p LEFT JOIN sakila.rental AS r ON p.rental_id = r.rental_id LEFT JOIN sakila.inventory AS i ON r.inventory_id = i.inventory_id LEFT JOIN sakila.film_actor AS fa ON i.film_id = fa.film_id LEFT JOIN sakila.actor AS a ON fa.actor_id = a.actor_id GROUP BY a.actor_id HAVING sales > 1000 ORDER BY sales DESC;

#List the customers who have rented more than 25 movies. Return their first names, last names, customer id, email, address line 1, postal code and the number of movies rented aliased as total_movies_rented. Order by first name, last name alphabeticlly.
SELECT c.first_name, c.last_name, r.customer_id, c.email, a.address, a.postal_code, COUNT(inventory_id) AS total_movies_rented FROM sakila.rental AS r LEFT JOIN sakila.customer AS c ON r.customer_id = c.customer_id LEFT JOIN sakila.address AS a ON c.address_id = a.address_id GROUP BY r.customer_id HAVING total_movies_rented > 25 ORDER BY c.first_name, c.last_name;

#List all the film categories and their average rental duration aliased as avg_rental_duration.
SELECT c.name, AVG(f.rental_duration) AS avg_rental_duration FROM sakila.film_category AS fc LEFT JOIN sakila.category AS c ON fc.category_id = c.category_id LEFT JOIN sakila.film AS f ON fc.film_id = f.film_id GROUP BY fc.category_id;
