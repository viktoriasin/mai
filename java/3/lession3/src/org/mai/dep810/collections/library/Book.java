package org.mai.dep810.collections.library;

public class Book implements Comparable<Book>{

    private int id;
    private String title;
    private String owner;

    public Book(Integer id, String title) throws IllegalArgumentException {
        if (id == null || title.trim().length() == 0 || title == null) {
            throw new IllegalArgumentException("Книга не может иметь нулевой идентификатор или название");
        }

        this.id = id;
        this.title = title;
        this.owner = null;
    }

    public String getOwner() {
        return this.owner;
    }

    public Integer getId() {
        return this.id;
    }

    public String getName() {
        return this.title;
    }

    public void setOwner(String o) {
        this.owner = o;
    }

    public void removeOwner() {
        this.owner = null;
    }

    boolean isFree() {
        return this.owner == null;
    }

    @Override
    public int compareTo(Book anotherBook) {
            return this.getName().compareTo(anotherBook.getName());
    }

}
