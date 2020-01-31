
--sqlplus.exe stack/stack@ORCL
create sequence SEQ_DIM_TAG;

select SEQ_DIM_TAG.nextval from dual;

insert into dim_tag(tag_id, tag_name)
select SEQ_DIM_TAG.nextval, tag_name from 
(select distinct trim(nvl(s.targettagname, t.tagname)) tag_name from tags t
left join tagsynonyms s
on t.tagname = s.sourcetagname);
commit;

select * from dim_tag;



-- create sequence SEQ_DIM_TAG_SYN;

-- select SEQ_DIM_TAG_SYN.nextval from dual;

-- insert into dim_tag_syn(tag_syn_id, tag_target_name, tag_src_name)
-- select SEQ_DIM_TAG_SYN.nextval, TARGETTAGNAME, SOURCETAGNAME from 
-- (select distinct TARGETTAGNAME, SOURCETAGNAME from TAGSYNONYMS);
-- commit;

-- select * from dim_tag_syn

--drop sequence SEQ_DIM_DATE
--create sequence SEQ_DIM_DATE;

--select SEQ_DIM_DATE.nextval from dual;
-- 

--CREATE TABLE dim_date (
--    date_id     INTEGER NOT NULL,
--    date_name   VARCHAR2(100 CHAR),
--    "LEVEL"     INTEGER
--);
--
--ALTER TABLE dim_date ADD CONSTRAINT dim_date_pk PRIMARY KEY ( date_id );


--проверка на уникальность
--select sourcetagname from TAGSYNONYMS
--group by sourcetagname
--having count(*) > 1

--проверка на то, что синонимы не будут множить строки, если соединять по sourcetagname
--select tagname from posts t
--join posttags pt on pt.postid = t.id
--join tags tt on tt.id = pt.tagid
-- join tagsynonyms s on trim(s.sourcetagname)=trim(tt.tagname)
--group by tagname,t.id
--having count(*) > 1




begin
  P_GENERATE_DIM_DATE(TO_DATE('10.10.10', 'DD.MM.YY'), TO_DATE('03.09.18', 'DD.MM.YY'));
end;

--rollback;

-- select max( TO_date(date_name, 'DD.MM.YY')) from DIM_DATE where "LEVEL"=1;

commit;

--drop table STG_FACT_POST purge;
create table STG_FACT_POST as
select distinct p.id post_id,
p.creationdate dat,
trim(nvl(s.targettagname, t.tagname)) tagname,
p.viewcount
from posts p
inner join posttags pt on (pt.postid = p.id)
inner join tags t on (t.id = pt.tagid)
left join  tagsynonyms s
on t.tagname = s.sourcetagname
where posttypeid=1;
-- select * from STG_FACT_POST;

-- select distinct STG_FACT_POST.dat from DIM_DATE D right join STG_FACT_POST
-- on TO_CHAR(STG_FACT_POST.dat, 'DD.MM.YY')  = D.date_name
-- where D.DATE_ID is null

--truncate table fact_post

create sequence SEQ_POST;
--drop sequence SEQ_POST

select SEQ_POST.nextval from dual;


--DROP TABLE fact_post CASCADE CONSTRAINTS;
insert into fact_post(
 post_id       ,
 DATE_ID,
    tag_id    	  ,
    viewcount     ,
    count         
)
select 
SEQ_POST.nextval,
(select DATE_ID from DIM_DATE D where D.DATE_NAME = TO_CHAR(STG_FACT_POST.dat, 'DD.MM.YY') and "LEVEL" = 1),
(select tag_id from dim_tag where dim_tag.tag_name = STG_FACT_POST.tagname) tag_id,
viewcount,
1
from STG_FACT_POST;

commit;


DROP TABLE fact_post_agg CASCADE CONSTRAINTS;
CREATE TABLE fact_post_agg (
    post_id       INTEGER ,
    date_id		  INTEGER ,
    tag_id    	  INTEGER,
    viewcount     INTEGER,
    count         INTEGER
);

insert into FACT_POST_AGG(
    post_id      ,
    date_id		 ,
    tag_id    	 ,
    viewcount    , 
    count         
)
select  null, 
date_id, 
tag_id, 
sum(viewcount), 
sum(count)
from fact_post
group by cube( date_id, tag_id)


select * from FACT_POST_AGG;


insert into fact_post(
    post_id       ,
    DATE_ID,
    tag_id        ,
    viewcount     ,
    count         
)
select 
SEQ_POST.nextval,
nvl((select DATE_ID from DIM_DATE D where D.DATE_NAME = TO_CHAR(STG_FACT_POST.dat, 'DD.MM.YY') and "LEVEL" = 1),-1),
nvl((select tag_id from dim_tag where dim_tag.tag_name = STG_FACT_POST.tagname),-1) tag_id,
viewcount,
count
from FACT_POST_AGG;



create materialized view MV_FACT_POST_AGG as
select   date_id, tag_id,  sum(viewcount), sum(count)
from fact_post
group by cube( date_id, tag_id)


begin
   dbms_mview.refresh('MV_FACT_POST_AGG');
end;

