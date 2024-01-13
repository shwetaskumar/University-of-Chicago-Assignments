USE crime_statistics;
# As the primary type 'CRIM SEXUAL ASSUALT' and 'CRIMINAL SEXUAL ASSUALT' are the same primary type category, the id for 'CRIM SEXUAL ASSUALT' was updated in the iucr table to 20 to reflect 'CRIMINAL SEXUAL ASSUALT' and tag them under the same category
UPDATE 
  iucr 
SET 
  primary_type_id = 20 
WHERE 
  primary_type_id = 19;

# Since NON - CRIMINAL, NON-CRIMINAL and NON - CRIMINAL  (SUBJECT SPECIFIED) are the same primary type category, the id of NON-CRIMINAL and NON - CRIMINAL  (SUBJECT SPECIFIED) was updated in the iucr table to 27 to reflect NON - CRIMINAL and tag them under the same category
UPDATE 
  iucr 
SET 
  primary_type_id = 27 
WHERE 
  primary_type_id IN (32, 34);

# iucr 5114 was a duplicate in the iucr table, preventing the table from from getting updated, hence it is being deleted
DELETE FROM 
  iucr 
WHERE 
  iucr = 5114 
  AND primary_type_id = 32;

# As the primary type 'NARCOTICS' and 'OTHER NARCOTIC VIOLATION' are the same primary type category, the id for 'OTHER NARCOTIC VIOLATION' was updated in the iucr table to 3 to reflect 'NARCOTICS' and tag them under the same category
UPDATE 
  iucr 
SET 
  primary_type_id = 3 
WHERE 
  primary_type_id = 31;

# Checking final results
SELECT 
  iucr, COUNT(iucr) AS c
FROM 
  iucr
  GROUP BY iucr
  HAVING c > 1; 

# Deleting the primary types just transformed
DELETE FROM 
  primary_type 
WHERE 
  primary_type_id IN (19, 31, 32, 34);

# Creating new column rank in iucr to define user defined safety scores.
ALTER TABLE `crime_statistics`.`iucr` 
ADD COLUMN `rank` INT NULL AFTER `description_id`;

#Creating a new schema to store the calculated scores for beat
CREATE SCHEMA crime_scores;

USE crime_scores;

DROP TABLE IF EXISTS scores_day;
CREATE TABLE crime_scores.scores_day (
  beat_id INT NOT NULL,
  sum_rank DECIMAL NOT NULL,
  normalized_score DECIMAL NOT NULL,
  safety_score DECIMAL NOT NULL,
  PRIMARY KEY (beat_id));
  
SELECT * FROM beat_scores;
SELECT * FROM beat_district_scores;
SELECT beat_id, AVG(safety_score) AS safety_score FROM beat_ward_scores GROUP BY beat_id;


SELECT * FROM crime_statistics.crime_report AS cr LEFT JOIN crime_statistics.iucr i ON cr.iucr = i.iucr;