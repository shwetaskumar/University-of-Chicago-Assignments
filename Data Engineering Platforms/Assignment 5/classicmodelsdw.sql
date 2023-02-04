/***********************************************
** File:   classicmodelsdw.sql
** Desc:   Classic Models Dimensional Schema
** Auth:   Ashish Pujari
** Date:   10/17/2019
** All Rights Reserved
************************************************/ 

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ALLOW_INVALID_DATES';
SET SQL_SAFE_UPDATES=0; 

-- -----------------------------------------------------
-- Schema classicmodelsdw
-- -----------------------------------------------------
DROP SCHEMA IF EXISTS classicmodelsdw;

CREATE SCHEMA IF NOT EXISTS `classicmodelsdw` DEFAULT CHARACTER SET utf8 ;
USE `classicmodelsdw` ;

-- -----------------------------------------------------
-- Table `classicmodelsdw`.`dimCustomers`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `classicmodelsdw`.`dimCustomers` (
    `customerNumber` INT(11) NOT NULL,
    `customerName` VARCHAR(50) NOT NULL,
    `contactLastName` VARCHAR(50) NOT NULL,
    `contactFirstName` VARCHAR(50) NOT NULL,
    `phone` VARCHAR(50) NOT NULL,
    `addressLine1` VARCHAR(50) NOT NULL,
    `addressLine2` VARCHAR(50) NULL DEFAULT NULL,
    `city` VARCHAR(50) NOT NULL,
    `state` VARCHAR(50) NULL DEFAULT NULL,
    `postalCode` VARCHAR(15) NULL DEFAULT NULL,
    `country` VARCHAR(50) NOT NULL,
    `salesRepEmployeeNumber` INT(11) NULL DEFAULT NULL,
    `creditLimit` DECIMAL(10 , 2 ) NULL DEFAULT NULL,
    PRIMARY KEY (`customerNumber`)
)  ENGINE=INNODB DEFAULT CHARACTER SET=UTF8;


-- -----------------------------------------------------
-- Table `classicmodelsdw`.`dimEmployees`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `classicmodelsdw`.`dimEmployees` (
    `employeeNumber` INT(11) NOT NULL,
    `lastName` VARCHAR(50) NOT NULL,
    `firstName` VARCHAR(50) NOT NULL,
    `extension` VARCHAR(10) NOT NULL,
    `email` VARCHAR(100) NOT NULL,
    `officeCode` VARCHAR(10) NOT NULL,
    `reportsTo` INT(11) DEFAULT NULL,
    `jobTitle` VARCHAR(50) NOT NULL,
    `city` VARCHAR(50) DEFAULT NULL,
    `state` VARCHAR(50) DEFAULT NULL,
    `country` VARCHAR(50) DEFAULT NULL,
    `postalCode` VARCHAR(15) DEFAULT NULL,
    `territory` VARCHAR(10) DEFAULT NULL,
    PRIMARY KEY (`employeeNumber`)
)  ENGINE=INNODB DEFAULT CHARACTER SET=UTF8;

-- -----------------------------------------------------
-- Table `classicmodelsdw`.`dimProducts`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `classicmodelsdw`.`dimProducts` (
    `productCode` VARCHAR(15) NOT NULL,
    `productName` VARCHAR(70) NOT NULL,
    `productLine` VARCHAR(50) NOT NULL,
    `productScale` VARCHAR(10) NOT NULL,
    `productVendor` VARCHAR(50) NOT NULL,
    `productDescription` TEXT NOT NULL,
    `buyPrice` DECIMAL(10 , 2 ) NOT NULL,
    `MSRP` DECIMAL(10 , 2 ) NOT NULL,
    PRIMARY KEY (`productCode`)
)  ENGINE=INNODB DEFAULT CHARACTER SET=UTF8;

-- -----------------------------------------------------
-- Table `classicmodelsdw`.`dimtime`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `classicmodelsdw`.`dimtime` (
    `dateId` BIGINT(20) NOT NULL,
    `date` DATE NOT NULL,
    `timestamp` BIGINT(20) NULL DEFAULT NULL,
    `weekend` CHAR(10) NOT NULL DEFAULT 'Weekday',
    `day_of_week` CHAR(10) NULL DEFAULT NULL,
    `month` CHAR(10) NULL DEFAULT NULL,
    `month_day` INT(11) NULL DEFAULT NULL,
    `year` INT(11) NULL DEFAULT NULL,
    `week_starting_monday` CHAR(2) NULL DEFAULT NULL,
    PRIMARY KEY (`dateId`),
    UNIQUE INDEX `date` (`date` ASC),
    INDEX `year_week` (`year` ASC , `week_starting_monday` ASC)
)  ENGINE=INNODB DEFAULT CHARACTER SET=UTF8;


-- -----------------------------------------------------
-- Table `classicmodelsdw`.`numbers`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `classicmodelsdw`.`numbers` (
    `number` BIGINT(20) NULL DEFAULT NULL
)  ENGINE=INNODB DEFAULT CHARACTER SET=UTF8;

-- -----------------------------------------------------
-- Table `classicmodelsdw`.`numbers_small`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `classicmodelsdw`.`numbers_small` (
    `number` INT(11) NULL DEFAULT NULL
)  ENGINE=INNODB DEFAULT CHARACTER SET=UTF8;

-- -----------------------------------------------------
-- Table `classicmodelsdw`.`factOrderDetails`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `classicmodelsdw`.`factOrderDetails` (
    `id` INT(11) NOT NULL AUTO_INCREMENT,
    `orderNumber` INT(11) NOT NULL,
    `productCode` VARCHAR(15) NOT NULL,
    `customerNumber` INT(11) NOT NULL,
    `employeeNumber` INT(11) NOT NULL,
    `orderDateId` BIGINT(20) NOT NULL,
    `requiredDateId` BIGINT(20) DEFAULT NULL,
    `shippedDateId` BIGINT(20) DEFAULT NULL,
    `quantityOrdered` INT(11) NULL,
    `quantityInStock` SMALLINT(6) NULL,
    `priceEach` DOUBLE NULL,
    `status` VARCHAR(15) NULL,
    PRIMARY KEY (`id`),
    INDEX `fk_factOrderDetails_dimProducts_idx` (`productCode` ASC),
    INDEX `fk_factOrderDetails_dimCustomers_idx` (`customerNumber` ASC),
    INDEX `fk_factOrderDetails_dimEmployees_idx` (`employeeNumber` ASC),
    INDEX `fk_factOrderDetails_dimTime_idx1` (`orderDateId` ASC),
    INDEX `fk_factOrderDetails_dimTime_idx2` (`requiredDateId` ASC),
    INDEX `fk_factOrderDetails_dimTime_idx3` (`shippedDateId` ASC),
    CONSTRAINT `fk_factOrderDetails_dimProducts` FOREIGN KEY (`productCode`)
        REFERENCES `classicmodelsdw`.`dimProducts` (`productCode`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_factOrderDetails_dimCustomers` FOREIGN KEY (`customerNumber`)
        REFERENCES `classicmodelsdw`.`dimCustomers` (`customerNumber`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_factOrderDetails_dimEmployees` FOREIGN KEY (`employeeNumber`)
        REFERENCES `classicmodelsdw`.`dimEmployees` (`employeeNumber`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_factOrderDetails_dimTime1` FOREIGN KEY (`orderDateId`)
        REFERENCES `classicmodelsdw`.`dimTime` (`dateId`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_factOrderDetails_dimTime2` FOREIGN KEY (`requiredDateId`)
        REFERENCES `classicmodelsdw`.`dimTime` (`dateId`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_factOrderDetails_dimTime3` FOREIGN KEY (`shippedDateId`)
        REFERENCES `classicmodelsdw`.`dimTime` (`dateId`)
        ON DELETE NO ACTION ON UPDATE NO ACTION
)  ENGINE=INNODB DEFAULT CHARACTER SET=UTF8;

-- -----------------------------------------------------
-- Populate Time dimension
-- -----------------------------------------------------

INSERT INTO numbers_small VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9);

INSERT INTO numbers
SELECT thousands.number * 1000 + hundreds.number * 100 + tens.number * 10 + ones.number
  FROM numbers_small thousands, numbers_small hundreds, numbers_small tens, numbers_small ones
LIMIT 1000000;

INSERT INTO dimTime (dateId, date)
SELECT number, DATE_ADD( '2014-01-01', INTERVAL number DAY )
  FROM numbers
  WHERE DATE_ADD( '2014-01-01', INTERVAL number DAY ) BETWEEN '2014-01-01' AND '2017-01-01'
  ORDER BY number;

UPDATE dimTime 
SET 
    timestamp = UNIX_TIMESTAMP(date),
    day_of_week = DATE_FORMAT(date, '%W'),
    weekend = IF(DATE_FORMAT(date, '%W') IN ('Saturday' , 'Sunday'),
        'Weekend',
        'Weekday'),
    month = DATE_FORMAT(date, '%M'),
    year = DATE_FORMAT(date, '%Y'),
    month_day = DATE_FORMAT(date, '%d');

UPDATE dimTime 
SET 
    week_starting_monday = DATE_FORMAT(date, '%v');

-- drop the temporary tables
DROP TABLE numbers;
DROP TABLE numbers_small;

-- -----------------------------------------------------
-- Copy Data from ClassicModels 
-- -----------------------------------------------------
# dimcustomers table 
# insert data into the dimcustomers table from classic models customers table 
INSERT INTO classicmodelsdw.dimcustomers
(SELECT * FROM classicmodels.customers);

# dimproducts table
# insert data into dimproducts table from classic models products table
INSERT INTO classicmodelsdw.dimproducts
(SELECT productCode, productName,productLine,
 productScale, productVendor, productDescription,
 buyPrice, MSRP FROM classicmodels.products);

# dimemployees table
# insert data into dimproducts table from classic models products table
INSERT INTO classicmodelsdw.dimemployees
(employeeNumber, lastName, firstName, 
extension, email, officeCode, reportsTo, jobTitle,
city, state, country, postalCode, territory)
SELECT employeeNumber, lastName, firstName, 
extension, email, offices.officeCode, reportsTo, jobTitle,
city, state, country, postalCode, territory 
FROM classicmodels.employees 
LEFT JOIN classicmodels.offices ON employees.officeCode = offices.officeCode;

# FactOrderDetails table
INSERT INTO classicmodelsdw.factorderdetails
(orderNumber, productCode, customerNumber, employeeNumber,
orderDateId, requiredDateId, shippedDateId, 
quantityOrdered, priceEach, quantityInStock, status)
SELECT orders.orderNumber, orderdetails.productCode, 
	orders.customerNumber, customers.salesRepEmployeeNumber,
	(SELECT dimTime.dateId FROM dimTime WHERE dimTime.date = orders.orderDate), 
    (SELECT dimTime.dateId FROM dimTime WHERE dimTime.date = orders.requiredDate), 
    (SELECT dimTime.dateId FROM dimTime WHERE dimTime.date = orders.shippedDate),     
    orderdetails.quantityOrdered, orderdetails.priceEach, 
    products.quantityInStock, orders.status
FROM classicmodels.orders, classicmodels.orderdetails, 
classicmodels.customers, classicmodels.products, dimTime
WHERE
orders.orderNumber = orderdetails.orderNumber 
AND orders.customerNumber = customers.customerNumber
AND orderdetails.productCode = products.productCode
AND dimTime.date = orders.orderDate;
