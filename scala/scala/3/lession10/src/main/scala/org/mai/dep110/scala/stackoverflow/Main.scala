package org.mai.dep110.scala.stackoverflow

object Main extends App {

  val loader = new DataLoader {
    override def basePath: String = "stackoverflow"
  }

  val entities = loader.loadData

  val cmts = Logic.getComments(entities)
  //cmts take(10) foreach println

  val (users, posts, comments, votes, badges, tags) = Logic.splitEntities(entities)
  //println(users.zip(posts).zip(comments).zip(votes).zip(badges).zip(tags).take(1).foreach(println))

  val reachPosts = Logic.enreachPosts(posts, users, tags);
  //reachPosts take(10) foreach println

  val reachComments = Logic.enreachComments(comments, posts, users)
  //reachComments take(10) foreach println

  val userLinks = Logic.findAllUserLinks(users)
  userLinks take(10) foreach println

  val topUsersByBadge = Logic.findTopUsersByBadge(users, badges, "Student", 100)
  //topUsersByBadge take(10) foreach println


}
