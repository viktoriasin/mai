package org.mai.dep810.collections.library;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.hamcrest.CoreMatchers.hasItem;
import static org.hamcrest.CoreMatchers.not;
import static org.junit.Assert.*;
import static org.junit.Assert.assertTrue;

public class LibraryImplTest {

    Library lib;
    Book bok = new Book(4,"X");
    Book k = new Book(11,"Exx");
    Book has_master = new Book(21, "Master");
    List<Book> l;


    @Before
    public void setUp() throws Exception {
        lib = new LibraryImpl();
//        lib.addNewBook(bok);
//        lib.addNewBook(new Book(1,"MN"));
//        lib.addNewBook(new Book(2,"IO"));
//        lib.addNewBook(new Book(3,"ZX"));
//
//        has_master.setOwner("Petya");
//        lib.addNewBook(has_master);

    }

    @After
    public void tearDown() throws Exception {
        lib = null;
    }

    @Test
    public void addNewBook(){

        lib.addNewBook(bok);

        assertThat(lib.findAvailableBooks(), hasItem(bok));
    }


    @Test(expected = IllegalArgumentException.class)
    public void addNewBook_alreadyExists() throws Exception {

        lib.addNewBook(bok);
        lib.addNewBook(bok);
//        lib.addNewBook(new Book(12, ""));
//        lib.addNewBook(new Book(11, null));
//        lib.addNewBook(new Book(1, "Dz"));

    }


    @Test
    public void borrowBook() {
        lib.addNewBook(bok);
        String stud = "Olya";
        lib.borrowBook(bok, stud);
        assertEquals(bok.getOwner(), stud);
    }

    @Test(expected = IllegalArgumentException.class)
    public void borrowBook_has_master() throws Exception  {

         has_master.setOwner("Petya");
         lib.addNewBook(has_master);
         String stud = "Olya";
         lib.borrowBook(has_master, stud);
        //lib.borrowBook(new Book(112,"New Book"), "Kolya");
    }

    @Test
    public void returnBook() {
        lib.addNewBook(bok);
        bok.setOwner("Petya");
        lib.returnBook(bok, "Petya");

        assertThat(lib.findAvailableBooks(), hasItem(bok));
        assertTrue(bok.isFree());
    }


    @Test(expected = IllegalArgumentException.class)
    public void returnBook_not_exists()  throws IllegalArgumentException  {
        lib.returnBook(new Book(987,"Buk"), "OS");
    }

    @Test(expected = IllegalArgumentException.class)
    public void returnBook_wrong_owner() throws IllegalArgumentException {
        lib.addNewBook(bok);
        lib.borrowBook(bok, "Maria");
        lib.returnBook(bok, "Petya");
    }

    @Test(expected = IllegalArgumentException.class)
    public void returnBook_isfree()  throws IllegalArgumentException  {
        lib.addNewBook(bok);
        lib.returnBook(bok, "Vladimir");
    }


    @Test
    public void findAvailableBooks() {
        l = lib.findAvailableBooks();
        for (Book b : l) {
            assertTrue(b.isFree());
        }
    }
}

