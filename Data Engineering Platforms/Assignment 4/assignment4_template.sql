########################## ASSIGNMENT 4a SQL ##############################

# Name: YOUR NAME HERE
# Date: CURRENT DATE

####### INSTRUCTIONS #######

# Read through the whole template and read each question carefully.  Make sure to follow all instructions.

# Each question should be answered with only one SQL query per question, unless otherwise stated.
# All queries must be written below the corresponding question number.
# Make sure to include the schema name in all table references (i.e. sakila.customer, not just customer)
# DO NOT modify the comment text for each question unless asked.
# Any additional comments you may wish to add to organize your code MUST be on their own lines and each comment line must begin with a # character
# If a question asks for specific columns and/or column aliases, they MUST be followed.
# Pay attention to the requested column aliases for aggregations and calculations. Otherwise, do not re-alias columns from the original column names in the tables unless asked to do so.
# Return columns in the order requested in the question.
# Do not concatenate columns together unless asked.

# Refer to the Sakila documentation for further information about the tables, views, and columns: https://dev.mysql.com/doc/sakila/en/

##########################################################################

## Desc: Joining Data, Nested Queries, Views and Indexes, Transforming Data

############################ PREREQUESITES ###############################

# These queries make use of the Sakila schema.  If you have issues with the Sakila schema, try dropping the schema and re-loading it from the scripts provided with Assignment 2.

# Run the following two SQL statements before beginning the questions:
SET SQL_SAFE_UPDATES=0;
UPDATE sakila.film SET language_id=6 WHERE title LIKE "%ACADEMY%";

############################### QUESTION 1 ###############################

# 1a) List the actors (first_name, last_name, actor_id) who acted in more then 25 movies.  Also return the count of movies they acted in, aliased as movie_count. Order by first and last name alphabetically.



# 1b) List the actors (first_name, last_name, actor_id) who have worked in German language movies. Order by first and last name alphabetically.



# 1c) List the actors (first_name, last_name, actor_id) who acted in horror movies and the count of horror movies by each actor.  Alias the count column as horror_movie_count. Order by first and last name alphabetically.



# 1d) List the customers who rented more than 3 horror movies.  Return the customer first and last names, customer IDs, and the horror movie rental count (aliased as horror_movie_count). Order by first and last name alphabetically.



# 1e) List the customers who rented a movie which starred Scarlett Bening.  Return the customer first and last names and customer IDs. Order by first and last name alphabetically.



# 1f) Which customers residing at postal code 62703 rented movies that were Documentaries?  Return their first and last names and customer IDs.  Order by first and last name alphabetically.



# 1g) Find all the addresses (if any) where the second address line is not empty and is not NULL (i.e., contains some text).  Return the address_id and address_2, sorted by address_2 in ascending order.



# 1h) List the actors (first_name, last_name, actor_id)  who played in a film involving a “Crocodile” and a “Shark” (in the same movie).  Also include the title and release year of the movie.  Sort the results by the actors’ last name then first name, in ascending order.



# 1i) Find all the film categories in which there are between 55 and 65 films. Return the category names and the count of films per category, sorted from highest to lowest by the number of films.  Alias the count column as count_movies. Order the results alphabetically by category.



# 1j) In which of the film categories is the average difference between the film replacement cost and the rental rate larger than $17?  Return the film categories and the average cost difference, aliased as mean_diff_replace_rental.  Order the results alphabetically by category.



# 1k) Create a list of overdue rentals so that customers can be contacted and asked to return their overdue DVDs. Return the title of the  film, the customer first name and last name, customer phone number, and the number of days overdue, aliased as days_overdue.  Order the results by first and last name alphabetically.
## NOTE: To identify if a rental is overdue, find rentals that have not been returned and have a rental date rental date further in the past than the film's rental duration (rental duration is in days)



# 1l) Find the list of all customers and staff for store_id 1.  Return the first and last names, as well as a column indicating if the name is "staff" or "customer", aliased as person_type.  Order results by first name and last name alphabetically.
## Note : use a set operator and do not remove duplicates



############################### QUESTION 2 ###############################

# 2a) List the first and last names of both actors and customers whose first names are the same as the first name of the actor with actor_id 8.  Order in alphabetical order by last name.
## NOTE: Do not remove duplicates and do not hard-code the first name in your query.



# 2b) List customers (first name, last name, customer ID) and payment amounts of customer payments that were greater than average the payment amount.  Sort in descending order by payment amount.
## HINT: Use a subquery to help



# 2c) List customers (first name, last name, customer ID) who have rented movies at least once.  Order results by first name, lastname alphabetically.
## Note: use an IN clause with a subquery to filter customers



# 2d) Find the floor of the maximum, minimum and average payment amount.  Alias the result columns as max_payment, min_payment, avg_payment.



############################### QUESTION 3 ###############################

# 3a) Create a view called actors_portfolio which contains the following columns of information about actors and their films: actor_id, first_name, last_name, film_id, title, category_id, category_name



# 3b) Describe (using a SQL command) the structure of the view.



# 3c) Query the view to get information (all columns) on the actor ADAM GRANT



# 3d) Insert a new movie titled Data Hero in Sci-Fi Category starring ADAM GRANT
## NOTE: If you need to use multiple statements for this question, you may do so.
## WARNING: Do not hard-code any id numbers in your where criteria.
## !! Think about how you might do this before reading the hints below !!
## HINT 1: Given what you know about a view, can you insert directly into the view? Or do you need to insert the data elsewhere?
## HINT 2: Consider using SET and LAST_INSERT_ID() to set a variable to aid in your process.



############################### QUESTION 4 ###############################

# 4a) Extract the street number (numbers at the beginning of the address) from the customer address in the customer_list view.  Return the original address column, and the street number column (aliased as street_number).  Order the results in ascending order by street number.
## NOTE: Use Regex to parse the street number



# 4b) List actors (first name, last name, actor id) whose last name starts with characters A, B or C.  Order by first_name, last_name in ascending order.
## NOTE: Use either a LEFT() or RIGHT() operator



# 4c) List film titles that contain exactly 10 characters.  Order titles in ascending alphabetical order.



# 4d) Return a list of distinct payment dates formatted in the date pattern that matches "22/01/2016" (2 digit day, 2 digit month, 4 digit year).  Alias the formatted column as payment_date.  Retrn the formatted dates in ascending order.



# 4e) Find the number of days each rental was out (days between rental_date & return_date), for all returned rentals.  Return the rental_id, rental_date, return_date, and alias the days between column as days_out.  Order with the longest number of days_out first.
