package org.mai.dep810.scala.stackoverflow

import scalikejdbc._

trait DBHelper {

  def db: Symbol

  def createTables = NamedDB(db) autoCommit { implicit session =>

    //create users table
    sql"""
         create table users(
          id Int primary key,
          display_name varchar(1000),
          location varchar(1000),
          reputation Int,
          views Int,
          up_votes Int,
          down_votes Int,
          account_id Int,
          creation_date timestamp,
          last_access_date timestamp
         )
       """.execute().apply()

    //TODO create posts table
    sql"""
         create table posts(
           id Int primary key,
            title varchar(1000),
            body varchar(50000),
            score Int,
            view_count Int,
            answer_count Int,
            comment_count Int,
            owner_user_id Int,
            last_editor_user_id Int,
            accepted_answer_id Int,
            creation_date timestamp,
            last_edit_date timestamp,
            last_activity_date timestamp
         )
       """.execute().apply()

    //TODO create commetns table
    sql"""
         create table commetns(
            id Int primary key,
            post_id Int,
            score Int,
            text varchar(10000),
            creation_date timestamp,
            user_id Int
         )
       """.execute().apply()
  }

  def dropTables = NamedDB(db) autoCommit { implicit session =>
    //drop users table
    sql"drop table if exists users".execute().apply()

    //drop posts table
    sql"drop table if exists posts".execute().apply()

    //create commetns table
    sql"drop table if exists comments".execute().apply()
  }

  def clearData = NamedDB(db) autoCommit { implicit session =>

    //delete from users
    sql"delete from users".execute().apply()

    //TODO delete from posts
    sql"delete from posts".execute().apply()

    //TODO delete from comments
    sql"delete from comments".execute().apply()

  }

  def saveData(users: Seq[User], posts: Seq[Post], comments: Seq[Comment]) = NamedDB(db) autoCommit {implicit session =>

    //save users
    val u = User.column

    users.foreach{ user =>
      withSQL(
        insert.into(User).namedValues(
          u.id -> user.id ,
          u.displayName -> user.displayName ,
          u.location -> user.location ,
          u.reputation -> user.reputation ,
          u.views -> user.views ,
          u.upVotes -> user.upVotes ,
          u.downVotes -> user.downVotes ,
          u.accountId -> user.accountId ,
          u.creationDate -> user.creationDate ,
          u.lastAccessDate -> user.lastAccessDate
        )
      ).update.apply()
    }

    //TODO save posts
    //save posts
    val p = Post.column

    posts.foreach{ post =>
      withSQL(
        insert.into(Post).namedValues(
          p.id -> post.id,
          p.title -> post.title,
          p.body -> post.body,
          p.score -> post.score,
          p.viewCount -> post.viewCount,
          p.answerCount -> post.answerCount,
          p.commentCount -> post.commentCount,
          p.ownerUserId -> post.ownerUserId,
          p.lastEditorUserId -> post.lastEditorUserId,
          p.acceptedAnswerId -> post.acceptedAnswerId,
          p.creationDate -> post.creationDate,
          p.lastEditDate -> post.lastEditDate,
          p.lastActivityDate -> post.lastActivityDate
        )
      ).update.apply()
    }


    //TODO save comments
    //save comments

    val c = Comment.column

    comments.foreach{ comment =>
      withSQL(
        insert.into(Comment).namedValues(
          c.id -> comment.id,
          c.postId -> comment.postId,
          c.score -> comment.score,
          c.text -> comment.text,
          c.creationDate -> comment.creationDate,
          c.userId -> comment.userId
        )
      ).update.apply()
    }
  }

  def extract(query: String): List[String] = NamedDB(db) readOnly { implicit session =>

    session
      .list(query){rs =>

        if(rs.row == 1) {
          List(
            (1 to rs.metaData.getColumnCount).map(i => rs.metaData.getColumnName(i)).mkString(","),
            (1 to rs.metaData.getColumnCount).map(i => if(rs.metaData.getColumnTypeName(i) == "VARCHAR") s""""${rs.string(i)}"""" else rs.string(i)).mkString(",")
          )
        } else {
          List(
            (1 to rs.metaData.getColumnCount).map(i => if(rs.metaData.getColumnTypeName(i) == "VARCHAR") s""""${rs.string(i)}"""" else rs.string(i)).mkString(",")
          )
        }
      }.flatten

  }

}


