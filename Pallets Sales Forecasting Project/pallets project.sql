CREATE DATABASE PALLETS 

USE pallets;

SELECT * 
FROM pallets_data; 



SELECT COUNT(*) AS num_rows
FROM pallets_data;

SELECT COUNT(*) AS num_columns
FROM pallets.columns;
WHERE TABLE_NAME ='Pallets_data'



SELECT MAX(RATE)
FROM Pallets_data;

SELECT MIN(RATE)
from pallets data
