package org.mai.dep110.scala.stackoverflow
import scala.util.matching.Regex
object Logic {

  //obtain all commetns from entities
  def getComments(entities: Seq[Entity]): Seq[Comment] = {
    entities.collect{ case p: Comment => p}
  }

  //split entities by type
  def splitEntities(entities: Seq[Entity]): (Seq[User], Seq[Post], Seq[Comment], Seq[Vote], Seq[Badge], Seq[Tag]) = {
    entities.foldLeft(Tuple6[Seq[User],Seq[Post], Seq[Comment], Seq[Vote], Seq[Badge], Seq[Tag]](Seq(),Seq(),Seq(),Seq(),Seq(),Seq()))
    {(accumulator, p) => (p match {
      case p : User => accumulator.copy(_1 = accumulator._1 :+ p)  //(accumulator._1 :+ p, accumulator._2,accumulator._3,accumulator._4,accumulator._5,accumulator._6)
      case p : Post => accumulator.copy(_2 = accumulator._2 :+ p)
      case p : Comment => accumulator.copy(_3 = accumulator._3 :+ p)
      case p : Vote => accumulator.copy(_4 = accumulator._4 :+ p)
      case p : Badge => accumulator.copy(_5 = accumulator._5 :+ p)
      case p : Tag => accumulator.copy(_6 = accumulator._6 :+ p)
   }
      )
    }


  }

  //populate fields owner, lastEditor, tags with particular users from Seq[Post] and tags from Seq[Tag]
  def enreachPosts(posts: Seq[Post], users: Seq[User], tags: Seq[Tag]): Seq[EnreachedPost] = {
    posts.map(p => EnreachedPost(p,
      users.find(u => u.id == p.id).orNull,
      users.find(u => u.id == p.lastEditorUserId).orNull,
      tags.filter(t => t.id == p.id)))
  }

//    def enreachPosts(posts: Seq[Post], users: Seq[User], tags: Seq[Tag]): Seq[EnreachedPost] = {
//      posts.map(p => EnreachedPost(p,
//        users.map {
//          case user: User if p.ownerUserId == user.id => user
//          case _ => null
//        }.find(user => user != null).orNull,
//        users.map {
//          case user: User if p.lastEditorUserId == user.id => user
//          case _ => null
//        }.find(user => user != null).orNull,
//        tags.filter(tag => {
//        tag match {
//          case tag: Tag if p.tags.contains(tag.tagName) => true
//          case _ => false
//        }
//    }
//   )
// )
//      )
//

  //populate fields post and owner with particular post from Seq[Post] and user from Seq[User]
  def enreachComments(comments: Seq[Comment],posts: Seq[Post], users: Seq[User]): Seq[EnreachedComment] = {
    comments.map(p => EnreachedComment(p,posts.find(u => u.id == p.postId).orNull, users.find(u => u.id == p.userId).orNull))

  }

  //find all links (like http://example.com/examplePage) in aboutMe field
  def findAllUserLinks(users: Seq[User]): Seq[(User, Seq[String])] = {
    val matcher: Regex = """"((http|ftp|https)://.*?\.(com|ru).*?)"""".r
        users.map(p => (p, matcher.findAllIn(p.about).toList))

  }

  //find all users with the reputation bigger then reputationLImit with particular badge
  def findTopUsersByBadge(users: Seq[User], badges: Seq[Badge], badgeName: String, reputationLimit: Int): Seq[User] = {
    users.collect{case a @ User( id: Int, _, _, _, reputation, _, _, _, _, _, _) if reputation > reputationLimit && badges.exists(p => p.userId == id && p.name== badgeName) => a}
  }

}

case class EnreachedPost(
                        post: Post,
                        owner: User,
                        lastEditor: User,
                        tags: Seq[Tag]
                        )

case class EnreachedComment(
                          comment: Comment,
                          post: Post,
                          owner: User
                        )
