package org.mai.dep810.library;


import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class LibraryImpl  implements Library{

    private String jdbcUrl;
    private String user;
    private String password;

    public LibraryImpl(String jdbcUrl, String user, String password) {
        this.jdbcUrl = jdbcUrl;
        this.user = user;
        this.password = password;
    }

    Connection getConnection() throws SQLException {
        return DriverManager.getConnection(jdbcUrl, user, password);
    }

    public void addNewBook(Book book) {
        String sqlAdd = "insert into BOOKS(book_id,book_title) values (? , ?)";

        try(Connection con  = getConnection();
            PreparedStatement checkStmt = con.prepareStatement(sqlAdd);) {
            checkStmt.setInt(1,book.getId());
            checkStmt.setString(2,book.getTitle());
            checkStmt.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void addAbonent(Student student) {
        String sqlAdd = "insert into ABONENTS(student_id,student_name) values (?, ?)";

        try(Connection con  = getConnection();
            PreparedStatement checkStmt = con.prepareStatement(sqlAdd);) {
            checkStmt.setInt(1,student.getId());
            checkStmt.setString(2,student.getName());
            checkStmt.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void borrowBook(Book book, Student student)  throws IllegalArgumentException {
        String sqlUpd = "update BOOKS set student_id = ? where book_id = ? and student_id is null";
        String checkSQL = "select 1 from BOOKS where book_id = ? and student_id is null";
        try (Connection con = getConnection();
                CallableStatement stmt = con.prepareCall(sqlUpd);
                PreparedStatement checkStmt = con.prepareStatement(checkSQL);) {
            checkStmt.setInt(1,book.getId());
            ResultSet rs = checkStmt.executeQuery();
            if (rs.next()) {
                stmt.setInt(1, student.getId());
                stmt.setInt(2, book.getId());
                int result = stmt.executeUpdate();
                //getConnection().commit();
            } else {
                throw new IllegalArgumentException("Книга уже занята другим пользователем.");

            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void returnBook(Book book, Student student)   throws IllegalArgumentException{
        String checkSql = "select 1 from BOOKS where student_id = ?  and book_id = ?";
        String sqlUpd = "update BOOKS set student_id = null where book_id = ?";
        try (   Connection con = getConnection();
                CallableStatement stmt_check = con.prepareCall(checkSql);
                CallableStatement stmt = con.prepareCall(sqlUpd)) {
            stmt_check.setInt(1,student.getId());
            stmt_check.setInt(2,book.getId());
            ResultSet rs = stmt_check.executeQuery();
            if (rs.next()) {
                stmt.setInt(1, book.getId());
                stmt.executeUpdate();
                getConnection().commit();
            } else {
                throw new IllegalArgumentException("Студент " + student.getName() + " не брал книгу " + book.getTitle());
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public List<Book> findAvailableBooks() {
        List<Book> boks = new ArrayList<>();
        String sqlStmt = "Select distinct book_id, book_title from BOOKS where student_id is null";
        try (Statement stmt = getConnection().createStatement();
             ResultSet rs = stmt.executeQuery(sqlStmt);) {
            while (rs.next()) {
                Integer id = rs.getInt("book_id");
                String title =  rs.getString("book_title");
                boks.add(new Book(id, title));
                System.out.println("Book available: " + id + " " + title);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return boks;
    }


    public List<Student> getAllStudents() {
        List<Student> std = new ArrayList<>();
        String sqlStmt = "Select distinct student_id, student_name from ABONENTS";
        try (Statement stmt = getConnection().createStatement();
             ResultSet rs = stmt.executeQuery(sqlStmt);) {
            while (rs.next()) {
                Integer id = rs.getInt("student_id");
                String name =  rs.getString("student_name");
                std.add(new Student(id, name));
                System.out.println("Student: " + id + " " + name);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return std;
    }
}

