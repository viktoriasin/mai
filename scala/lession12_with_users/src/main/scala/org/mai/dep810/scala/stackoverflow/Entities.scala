package org.mai.dep810.scala.stackoverflow

import java.time.LocalDateTime

import scalikejdbc._

abstract class Entity(id: Int)

case class Post(
                 id: Int,
                 title: String,
                 body: String,
                 score: Int,
                 viewCount: Int,
                 answerCount: Int,
                 commentCount: Int,
                 ownerUserId: Int,
                 lastEditorUserId: Int,
                 acceptedAnswerId: Int,
                 creationDate: LocalDateTime,
                 lastEditDate: LocalDateTime,
                 lastActivityDate: LocalDateTime,
                 tags: Seq[String]) extends Entity(id)

case class Comment(
                    id: Int,
                    postId: Int,
                    score: Int,
                    text: String,
                    creationDate: LocalDateTime,
                    userId: Int) extends Entity(id)


case class User(
                 id: Int,
                 displayName: String,
                 location: String,
                 reputation: Int,
                 views: Int,
                 upVotes: Int,
                 downVotes: Int,
                 accountId: Int,
                 creationDate: LocalDateTime,
                 lastAccessDate: LocalDateTime) extends Entity(id)

object User extends SQLSyntaxSupport[User] {
  override def connectionPoolName: Any = stackOverflowDB

  override def tableName: String = "users"

  def apply(u: ResultName[User])(rs: WrappedResultSet): Unit = {
    new User(
      rs.int(u.id),
      rs.string(u.displayName),
      rs.string(u.location),
      rs.int(u.reputation),
      rs.int(u.views),
      rs.int(u.upVotes),
      rs.int(u.downVotes),
      rs.int(u.accountId),
      rs.localDateTime(u.creationDate),
      rs.localDateTime(u.lastAccessDate)
    )
  }
}


object Post extends SQLSyntaxSupport[Post] {
  override def connectionPoolName: Any = stackOverflowDB

  override def tableName: String = "posts"

  def apply(post: ResultName[Post])(rs: WrappedResultSet): Unit = {
    new Post(
      rs.int(post.id),
      rs.string(post.title),
      rs.string(post.body),
      rs.int(post.score),
      rs.int(post.viewCount),
      rs.int(post.answerCount),
      rs.int(post.commentCount),
      rs.int(post.ownerUserId),
      rs.int(post.lastEditorUserId),
      rs.int(post.acceptedAnswerId),
      rs.localDateTime(post.creationDate),
      rs.localDateTime(post.lastEditDate),
      rs.localDateTime(post.lastActivityDate),
      Seq()
    )
  }
}
object Comment extends SQLSyntaxSupport[Comment] {
  override val tableName: String = "comments"

  override def connectionPoolName: Any = stackOverflowDB

  def apply(comment: ResultName[Comment])(rs: WrappedResultSet): Comment = {
    new Comment(
      rs.int(comment.id),
      rs.int(comment.postId),
      rs.int(comment.score),
      rs.string(comment.text),
      rs.localDateTime(comment.creationDate),
      rs.int(comment.userId)
    )
  }
}

case class Config (
                    commandLoad: String = "",
                    commandClean: String = "",
                    commandInit: String = "",
                    commandExtract: String = "",
                    path: String = "",
                    file: String = "",
                    append: Boolean = false,
                    dropTables: Boolean = false,
                    forse: Boolean = false,
                    query: String = ""
                  )