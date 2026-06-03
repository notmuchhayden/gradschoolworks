package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

class BookManager {
    private final List<String> books = new ArrayList<>();

    public void addBook(String book) {
        books.add(book);
    }

    public boolean borrow(String person, String book) {
        if (books.remove(book)) {
            return true;
        }
        return false;
    }

    public int getBookCount() {
        return books.size();
    }
}



class BorrowerManager {
    private final List<String> borrowers = new ArrayList<>();

    public void borrow(String person, String book) {
        borrowers.add(person + ":" + book);
    }

    public int getBorrowerCount() {
        return borrowers.size();
    }
}


class LibraryInfoManager {
    private final String libraryId;
    private String librarian;

    public LibraryInfoManager(String libraryId, String librarian) {
        this.libraryId = libraryId;
        this.librarian = librarian;
    }

    public void renameLibrarian(String name) {
        librarian = name;
    }

    public String getLibraryInfo() {
        return libraryId + ":" + librarian;
    }
}


public class LargeClass06_qwt1 {
    private final LibraryInfoManager libraryInfoManager;
    private final BookManager bookManager;
    private final BorrowerManager borrowerManager;
    private int lostCount;
    private int damagedCount;
    private boolean open;

    public LargeClass06_qwt1(String libraryId, String librarian) {
        this.libraryInfoManager = new LibraryInfoManager(libraryId, librarian);
        this.bookManager = new BookManager();
        this.borrowerManager = new BorrowerManager();
        this.open = true;
    }

    public void addBook(String book) {
        if (open) {
            bookManager.addBook(book);
        }
    }

    public void borrow(String person, String book) {
        if (open && bookManager.borrow(person, book)) {
            borrowerManager.borrow(person, book);
        }
    }

    public void markLost() {
        lostCount++;
    }

    public void markDamaged() {
        damagedCount++;
    }

    public String inventorySummary() {
        return libraryInfoManager.getLibraryInfo() + ":" + bookManager.getBookCount() + ":" + borrowerManager.getBorrowerCount() + ":" + lostCount + ":" + damagedCount;
    }
}

