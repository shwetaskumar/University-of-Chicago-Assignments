SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema rev (real estate vigilantes)
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema rev
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `rev` DEFAULT CHARACTER SET utf8 ;
USE `rev` ;

-- -----------------------------------------------------
-- Table `rev`.`primary_type`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`primary_type` (
  `primary_type_id` INT NOT NULL AUTO_INCREMENT,
  `primary_type_desc` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`primary_type_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`description`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`description` (
  `description_id` INT NOT NULL AUTO_INCREMENT,
  `description` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`description_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`IUCR`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`iucr` (
  `iucr` INT NOT NULL,
  `primary_type_primary_type_id` INT NOT NULL,
  `description_description_id` INT NOT NULL,
  PRIMARY KEY (`primary_type_primary_type_id`, `description_description_id`, `iucr`),
  INDEX `fk_iucr_description1_idx` (`description_description_id` ASC) VISIBLE,
  CONSTRAINT `fk_iucr_description1`
    FOREIGN KEY (`description_description_id`)
    REFERENCES `rev`.`description` (`description_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`district`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`district` (
  `district_id` INT NOT NULL,
  `district_desc` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`district_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`ward`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`ward` (
  `ward_id` INT NOT NULL,
  `ward_desc` VARCHAR(45) NOT NULL,
  `district_district_id` INT NOT NULL,
  PRIMARY KEY (`ward_id`, `district_district_id`),
  INDEX `fk_ward_district1_idx` (`district_district_id` ASC) VISIBLE,
  CONSTRAINT `fk_ward_district1`
    FOREIGN KEY (`district_district_id`)
    REFERENCES `rev`.`district` (`district_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`beat`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`beat` (
  `beat_id` INT NOT NULL,
  `ward_ward_id` INT NOT NULL,
  PRIMARY KEY (`beat_id`, `ward_ward_id`),
  INDEX `fk_beat_ward1_idx` (`ward_ward_id` ASC) VISIBLE,
  CONSTRAINT `fk_beat_ward1`
    FOREIGN KEY (`ward_ward_id`)
    REFERENCES `rev`.`ward` (`ward_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`community`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`community` (
  `community_id` INT NOT NULL,
  `community_name` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`community_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`location`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`location` (
  `location_id` INT NOT NULL AUTO_INCREMENT,
  `block` VARCHAR(100) NOT NULL,
  `latitude` DOUBLE NULL,
  `longitude` DOUBLE NULL,
  `beat_beat_id` INT NOT NULL,
  `community_community_id` INT NOT NULL,
  PRIMARY KEY (`location_id`, `beat_beat_id`, `community_community_id`),
  INDEX `fk_location_beat1_idx` (`beat_beat_id` ASC) VISIBLE,
  INDEX `fk_location_community1_idx` (`community_community_id` ASC) VISIBLE,
  CONSTRAINT `fk_location_beat1`
    FOREIGN KEY (`beat_beat_id`)
    REFERENCES `rev`.`beat` (`beat_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_location_community1`
    FOREIGN KEY (`community_community_id`)
    REFERENCES `rev`.`community` (`community_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`crime_location`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`crime_location` (
  `crime_location_id` INT NOT NULL AUTO_INCREMENT,
  `crime_location_description` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`crime_location_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `rev`.`crime_report`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rev`.`crime_report` (
  `ID` INT NOT NULL AUTO_INCREMENT,
  `case_number` VARCHAR(45) NOT NULL,
  `date` DATETIME NOT NULL,
  `arrest` BINARY(1) NOT NULL,
  `domestic` BINARY(1) NOT NULL,
  `updated_on` TIMESTAMP NULL,
  PRIMARY KEY (`ID`)
  )
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
