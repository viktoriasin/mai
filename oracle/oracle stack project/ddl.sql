select min(creationdate) mindat,  max(creationdate) maxdat from posts
where posttypeid = 1

--10.10.10	02.09.18

DROP TABLE fact_post CASCADE CONSTRAINTS;

DROP TABLE dim_date CASCADE CONSTRAINTS;


DROP TABLE dim_tag CASCADE CONSTRAINTS;


CREATE TABLE dim_date (
    date_id     INTEGER NOT NULL,
    date_name   VARCHAR2(100 CHAR),
    "LEVEL"     INTEGER
);

ALTER TABLE dim_date ADD CONSTRAINT dim_date_pk PRIMARY KEY ( date_id );


CREATE TABLE dim_tag (
    tag_id      INTEGER NOT NULL,
    tag_name    VARCHAR2(1000 CHAR)
);

ALTER TABLE dim_tag ADD CONSTRAINT dim_tag_pk PRIMARY KEY ( tag_id );


CREATE TABLE fact_post (
    post_id       INTEGER NOT NULL,
    date_id		  INTEGER not null,
    tag_id    	  INTEGER,
    viewcount     INTEGER,
    count         INTEGER
);

ALTER TABLE fact_post ADD CONSTRAINT fact_post_pk PRIMARY KEY ( post_id );


ALTER TABLE fact_post
    ADD CONSTRAINT fact_post_dim_tag_fk FOREIGN KEY ( tag_id )
        REFERENCES dim_tag ( tag_id );



ALTER TABLE fact_post
    ADD CONSTRAINT fact_post_dim_date_fk FOREIGN KEY ( date_id )
        REFERENCES dim_date ( date_id );