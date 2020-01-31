package org.mai.dep810.library;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.hasItems;

public class LibraryTest {

    LibraryImpl library;
    List<Book> l;
    List<Student> s;

    @Before
    //Run before each test
    public void setUp() throws Exception {
        library = new LibraryImpl("jdbc:h2:mem:library", "", "");
        Connection connection = library.getConnection();
        try (Statement stmp = connection.createStatement()) {
            String tablesql = "CREATE TABLE IF NOT EXISTS ABONENTS( " +
                    "    student_id int, " +
                    "    student_name varchar(255), " +
                    "    primary key (student_id) " +
                    "); " +
                    " " +
                    "CREATE TABLE IF NOT EXISTS BOOKS( " +
                    "    book_id int, " +
                    "    book_title varchar(255), " +
                    "    student_id int, " +
                    "    primary key (book_id) " +
                    ");";
            stmp.execute(tablesql);
        } catch (Exception e) {
            System.out.println("ex");
        }


    }

    @After
    //Run after each test
    public void tearDown() throws Exception {
        Connection connection = library.getConnection();
        try (Statement stmp = connection.createStatement()) {
            stmp.execute("DROP TABLE BOOKS");
            stmp.execute(("DROP TABLE ABONENTS"));
        }
    }

    @Test
    public void addNewBook() throws Exception {

        Book b = new Book(1, "Book one");
        library.addNewBook(b);

        Connection connection = library.getConnection();
        PreparedStatement stmt = connection.prepareStatement("select * from BOOKS where book_id = ?");
        stmt.setInt(1, b.getId());

        ResultSet rs = stmt.executeQuery();
        assertTrue(rs.next());

        int id = rs.getInt("book_id");
        assertEquals(b.getId(), id);

        String title = rs.getString("book_title");
        assertEquals(b.getTitle(), title);

        int abonentId = rs.getInt("student_id");
        assertTrue(rs.wasNull());

        assertTrue(!rs.next());

    }

    @Test
    public void addAbonent() throws Exception {
        Student s = new Student(1, "Vasya");
        library.addAbonent(s);
        try (
                Connection connection = library.getConnection();
                PreparedStatement stmt = connection.prepareStatement("select * from ABONENTS where student_id = ?");) {
            stmt.setInt(1, s.getId());

            ResultSet rs = stmt.executeQuery();
            assertTrue(rs.next());

            int id = rs.getInt("student_id");
            assertEquals(s.getId(), id);

            String name = rs.getString("student_name");
            assertEquals(s.getName(), name);

            assertTrue(!rs.next());
        }
    }
    @Test
    public void borrowBook() throws Exception {
        Student s = new Student(2, "Olya");
        library.addAbonent(s);

        Book b = new Book(2, "Book two");
        library.addNewBook(b);

        library.borrowBook(b, s);
        try (Connection connection = library.getConnection();
             PreparedStatement stmt = connection.prepareStatement("select book_id, book_title, student_id from BOOKS where book_id = ?");) {

            stmt.setInt(1, b.getId());
            ResultSet rs = stmt.executeQuery();
            assertTrue(rs.next());

            int id = rs.getInt("student_id");
            assertEquals(s.getId(), id);


            assertTrue(!rs.next());
        }
    }

    @Test
    public void returnBook() throws Exception {
        Student s = new Student(3, "Olya");
        library.addAbonent(s);

        Book b = new Book(3, "Book two");
        library.addNewBook(b);
        library.borrowBook(b, s);
        library.returnBook(b,s);

        try (Connection connection = library.getConnection();
             PreparedStatement stmt = connection.prepareStatement("select book_id, book_title, student_id from BOOKS where book_id = ?");) {

            stmt.setInt(1, b.getId());
            ResultSet rs = stmt.executeQuery();
            assertTrue(rs.next());

            int id = rs.getInt("student_id");
            assertTrue(rs.wasNull());

            assertTrue(!rs.next());
        }

    }

    @Test
    public void findAvailableBooks() throws Exception {
        Book b4 = new Book(4, "Book  four");
        Book b5 = new Book(5, "Book  five");
        List expected = new ArrayList<>();
        expected.add(b4);
        expected.add(b5);
        library.addNewBook(b4);
        library.addNewBook(b5);
        l = library.findAvailableBooks();
        assertThat(l, hasItems(b4,b5));



    }

    @Test
    public void getAllStudents() throws Exception {

        Student b4 = new Student(4, "Student  four");
        Student b5 = new Student(5, "Student  five");
        List expected = new ArrayList<>();
        expected.add(b4);
        expected.add(b5);
        library.addAbonent(b4);
        library.addAbonent(b5);
        s = library.getAllStudents();
        assertThat(s, hasItems(b4,b5));
    }

    @Test(expected = IllegalArgumentException.class)
    public void borrowBook_has_master() throws Exception  {
        Book b4 = new Book(4, "Book  four");
        Student s = new Student(5,"Vasya");
        Student s_fail = new Student(6,"Olya");
        library.addAbonent(s);
        library.addNewBook(b4);
        library.borrowBook(b4,s);
        library.borrowBook(b4,s_fail);
    }


    @Test(expected = IllegalArgumentException.class)
    public void returnBook_wrong_owner() throws IllegalArgumentException {
        Book b4 = new Book(4, "Book  four");
        Student s = new Student(5,"Vasya");
        Student s_fail = new Student(6,"Olya");
        library.addAbonent(s);
        library.addNewBook(b4);
        library.borrowBook(b4,s);
        library.returnBook(b4,s_fail);
    }
}