
#EDA
#MEAN
SELECT AVG(RATE) AS mean_value   
FROM pallets_data;

SELECT AVG(QUANTITY) AS mean_value
FROM pallets_data;


#MEDAIN
SELECT AVG(median_value) AS median
FROM (
  SELECT QUANTITY AS median_value
  FROM pallets_data
  ORDER BY QUANTITY
  LIMIT 2
 ) AS median_subquery;

#MODE
SELECT STATE AS mode, COUNT(*) AS frequency
FROM  pallets_data
GROUP BY STATE
HAVING COUNT(*) = (
    SELECT MAX(mode_count)
    FROM (
        SELECT COUNT(*) AS mode_count
        FROM pallets_data
        GROUP BY STATE
    ) AS counts
)

#VARIANCE
SELECT VARIANCE(QUANTITY) AS variance
FROM pallets_data;

#STANDARD DEVIATION
SELECT SQRT(variance_value) AS standard_deviation
FROM (
  SELECT VARIANCE(QUANTITY) AS variance_value
  FROM pallets_data
) AS variance_subquery;

#RANGE
SELECT MIN(RATE)
FROM pallets_data
SELECT MAX(RATE)
FROM pallets_data


#SKEWNESS
SELECT (SUM(POW(RATE- 1623.2611, 3)) / (COUNT(*) * POW(STD(RATE), 3))) AS skewness
FROM pallets_data
#KURTOSIS
SELECT ((COUNT(*) * SUM(POW(RATE - 1623.2611, 4))) / (POW(STD(RATE), 4))) - 3 AS kurtosis
FROM pallets_data





#DATA PREPROCESSING
#OUTLIER ANALYSIS


SELECT RATE, QUANTITY
FROM pallets_data
ORDER BY QUANTITY ASC;





#HANDLING MISSING VALUES
SELECT * FROM pallets_data WHERE STATE IS NOT NULL;

SELECT U_TRINPD , IFNULL(U_TRINPD, 0) AS filled_value FROM pallets_data;

SELECT U_Frt, IFNULL(U_Frt, (SELECT AVG(U_Frt) FROM pallets_data)) AS filled_value FROM pallets_data;






#HANDLING CATEGORICAL VARIABLES
SELECT
    CASE WHEN U_AssetClass = 'Category 1' THEN 1 ELSE 0 END AS category_1
    FROM pallets_data;

SELECT
    U_DocStatus,
    ROW_NUMBER() OVER (ORDER BY U_DocStatus) AS encoded_category
FROM pallets_data;




