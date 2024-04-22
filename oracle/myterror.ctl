OPTIONS (SKIP=1)
LOAD DATA
INFILE 'myterror.csv'
INTO TABLE MYTERROR
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
        eventid      CHAR,
        iyear        INTEGER,
        imonth       INTEGER,
        iday         INTEGER,
        country      INTEGER,
        country_txt  CHAR,
        region       INTEGER,
        region_txt   CHAR,
        provstate    CHAR,
        city         CHAR,
        latitude     FLOAT,
        longitude    FLOAT
)
