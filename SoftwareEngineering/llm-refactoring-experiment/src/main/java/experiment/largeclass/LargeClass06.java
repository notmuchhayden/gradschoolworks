package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass06 {
    private final String libraryId;
    private String librarian;
    private final List<String> books = new ArrayList<>();
    private final List<String> borrowers = new ArrayList<>();
    private int lostCount;
    private int damagedCount;
    private boolean open;

    public LargeClass06(String libraryId, String librarian) {
        this.libraryId = libraryId;
        this.librarian = librarian;
        this.open = true;
    }

    public void addBook(String book) {
        if (open) {
            books.add(book);
        }
    }

    public void borrow(String person, String book) {
        if (open && books.remove(book)) {
            borrowers.add(person + ":" + book);
        }
    }

    public void markLost() {
        lostCount++;
    }

    public void markDamaged() {
        damagedCount++;
    }

    public String inventorySummary() {
        return libraryId + ":" + librarian + ":" + books.size() + ":" + borrowers.size() + ":" + lostCount + ":" + damagedCount;
    }

    public void renameLibrarian(String name) {
        librarian = name;
    }
}
