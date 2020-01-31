package org.mai.dep810.collections.library;

import java.util.*;
import java.util.stream.Collectors;


public class LibraryImpl implements Library{

    //private List<Book> books = new ArrayList<Book>();
    //private Set<Integer> ids = new HashSet<>();

    private Map<Integer, Book> bks = new HashMap<>();
    /* Регистрация новой книги */
    @Override
    public void addNewBook(Book book) throws IllegalArgumentException{
        if (bks.containsKey(book.getId())) {
            throw new IllegalArgumentException("ID добавляемой книги уже присутствует в библиотеке.");
        }
        bks.put(book.getId(), book);
    }

    /* Студент берет книгу */
    @Override
    public void borrowBook(Book book, String student) throws IllegalArgumentException {
        if (!bks.containsKey(book.getId())) {
            throw new IllegalArgumentException("Книга '" + book.getName() + "' не представлена в библиотеке.");
        }

        if (!book.isFree()) {
            throw new IllegalArgumentException("Книга уже занята другим пользователем.");
        }

        book.setOwner(student);
    }

    /* Студент возвращает книгу */
    @Override
    public void returnBook(Book book, String student) {
        if (!bks.containsKey(book.getId())) {
            throw new IllegalArgumentException("Книга '" + book.getName() + "' не представлена в библиотеке.");
        }
        if (book.isFree()) {
            throw  new IllegalArgumentException("Книга сводна и не была никем занята.");
        }
        if (!book.getOwner().equals(student)) {
            throw  new IllegalArgumentException("Книга занята другим студентом.");
        }
        book.removeOwner();
    };

    /* Получить список свободных книг */
    @Override
    public List<Book> findAvailableBooks() {
        List<Book> b = bks.values().stream().filter(Book::isFree).sorted().collect(Collectors.toList());

        return b;
    };
}



