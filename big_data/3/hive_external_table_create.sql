
--создание внешних таблиц в hive


-- Id,Reputation,CreationDate,DisplayName,LastAccessDate,WebsiteUrl,Location,AboutMe,Views,UpVotes,DownVotes,ProfileImageUrl,AccountId,Age
CREATE EXTERNAL TABLE users(
	Id BIGINT,  
	Reputation INT, 
	CreationDate STRING,  
	DisplayName STRING, 
	LastAccessDate STRING, 
	WebsiteUrl STRING,
	Location STRING, 
	AboutMe STRING,
	Views INT, 
	UpVotes INT, 
	DownVotes INT, 
	ProfileImageUrl STRING,
	AccountId INT,
	Age INT
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\001' 
	STORED AS SEQUENCEFILE
	LOCATION '/user/stud/stackoverflow/master/Users';



CREATE EXTERNAL TABLE posts_scifi (
	Id bigint, 
	PostTypeId int, 
	AcceptedAnswerId bigint, 
	ParentId bigint, 
	CreationDate string, 
	DeletionDate string, 
	Score bigint, 
	ViewCount bigint, 
	OwnerUserId bigint, 
	OwnerDisplayName string, 
	LastEditorUserId bigint, 
	LastEditorDisplayName string, 
	LastEditDate string, 
	LastActivityDate string, 
	Title string, 
	Tags string, 
	AnswerCount int, 
	CommentCount int, 
	FavoriteCount int, 
	ClosedDate string, 
	CommunityOwnedDate string 
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\u0001' 
	STORED AS SEQUENCEFILE 
	LOCATION '/user/stud/stackoverflow/master/scifi/Posts';
	
CREATE EXTERNAL TABLE posts_sample_scifi (
	Id bigint, 
	PostTypeId int, 
	AcceptedAnswerId bigint, 
	ParentId bigint, 
	CreationDate string, 
	DeletionDate string, 
	Score bigint, 
	ViewCount bigint, 
	OwnerUserId bigint, 
	OwnerDisplayName string, 
	LastEditorUserId bigint, 
	LastEditorDisplayName string, 
	LastEditDate string, 
	LastActivityDate string, 
	Title string, 
	Tags string, 
	AnswerCount int, 
	CommentCount int, 
	FavoriteCount int, 
	ClosedDate string, 
	CommunityOwnedDate string 
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\u0001' 
	STORED AS SEQUENCEFILE 
	LOCATION '/user/stud/stackoverflow/master/scifi/posts_sample';

	
-- Id,UserId,Name,Date,Class,TagBased
CREATE EXTERNAL TABLE badges_scifi (
	Id bigint, 
	UserId bigint, 
	Name string, 
	Date string, 
	Class int, 
	TagBased string
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\u0001' 
	STORED AS SEQUENCEFILE 
	LOCATION '/user/stud/stackoverflow/master/scifi/Badges';

-- Id,PostId,Score,CreationDate,UserDisplayName,UserId
CREATE EXTERNAL TABLE comments_scifi (
	Id bigint, 
	PostId bigint, 
	Score int, 
	CreationDate string,
	UserDisplayName string,
	UserId bigint
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\u0001' 
	STORED AS SEQUENCEFILE 
	LOCATION '/user/stud/stackoverflow/master/scifi/Comments';
	
-- Id,CreationDate,PostId,RelatedPostId,LinkTypeId
CREATE EXTERNAL TABLE postlinks_scifi (
	Id bigint, 
	CreationDate string,
	PostId bigint, 
	RelatedPostId bigint, 
	LinkTypeId int
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\u0001' 
	STORED AS SEQUENCEFILE 
	LOCATION '/user/stud/stackoverflow/master/scifi/PostLinks';
	
-- Id,TagName,Count,ExcerptPostId,WikiPostId
CREATE EXTERNAL TABLE tags_scifi (
	Id bigint,
	TagName	string,
	Count int,
	ExcerptPostId bigint, 
	WikiPostId bigint
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\u0001' 
	STORED AS SEQUENCEFILE 
	LOCATION '/user/stud/stackoverflow/master/scifi/Tags';
	
	
-- Id,PostId,VoteTypeId,UserId,CreationDate,BountyAmount
CREATE EXTERNAL TABLE votes_scifi (
	Id bigint,
	PostId bigint,
	VoteTypeId int,
	UserId bigint,
	CreationDate string,
	BountyAmount int
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\u0001' 
	STORED AS SEQUENCEFILE 
	LOCATION '/user/stud/stackoverflow/master/scifi/Votes';

