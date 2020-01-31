--1

select id, name, SYS_CONNECT_BY_PATH(case when name = '/' then '' else name end, '/') Path
from FILE_SYSTEM
START WITH PARENT_ID is null 
   CONNECT BY NOCYCLE PRIOR id = PARENT_ID
   order by level;
   

--2 
 
 with h as (
select id, name,prior name parent_name,  SYS_CONNECT_BY_PATH(case when name = '/' then '' else name end, '/') Path, file_size, type
from FILE_SYSTEM
START WITH PARENT_ID is null 
   CONNECT BY NOCYCLE PRIOR id = PARENT_ID

   )  
select id
, name
, path
, nvl((select sum(file_size) from h  where h.path like t.path || '%' and h.type='FILE'),0) total_size --instr(h.path, t.path) > 0
from (
select id
, name
, level lvl
, prior name parent_name
, SYS_CONNECT_BY_PATH(case when name = '/' then '' else name end, '/') Path, file_size, type
from FILE_SYSTEM
START WITH PARENT_ID is null 
   CONNECT BY NOCYCLE PRIOR id = PARENT_ID
   ) t
   where type = 'DIR'



--3 

 with h as (
select id, name,prior name parent_name,  SYS_CONNECT_BY_PATH(case when name = '/' then '' else name end, '/') Path, file_size, type
from FILE_SYSTEM
START WITH PARENT_ID is null 
   CONNECT BY NOCYCLE PRIOR id = PARENT_ID
   )  
   select id
   , name
   , path
   , total_size
   , lvl
   , round(ratio_to_report(total_size) over(partition by lvl),3)
   from (
select id
, name
, path
, file_size
, type
, parent_name
, lvl
, nvl((select sum(file_size) from h  where h.path like t.path || '%' and h.type='FILE'),0) total_size --instr(h.path, t.path) > 0
from (
select id
, name
, level lvl
, prior name parent_name
, SYS_CONNECT_BY_PATH(case when name = '/' then '' else name end, '/') Path, file_size, type
from FILE_SYSTEM
START WITH PARENT_ID is null 
   CONNECT BY NOCYCLE PRIOR id = PARENT_ID
   ) t
   where type = 'DIR'
)
order by lvl
