package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

class LoanManager {
    private final List<String> borrowers = new ArrayList<>();

    public void addLoan(String person, String book) {
        borrowers.add(person + ":" + book);
    }

    public int getLoanCount() {
        return borrowers.size();
    }
}

class LibraryStatistics {
    private int lostCount = 0;
    private int damagedCount = 0;

    public void markLost() {
        lostCount++;
    }

    public void markDamaged() {
        damagedCount++;
    }

    public int getLostCount() {
        return lostCount;
    }

    public int getDamagedCount() {
        return damagedCount;
    }
}

class BookInventory {
    private final List<String> books = new ArrayList<>();

    public void addBook(String book) {
        books.add(book);
    }

    public boolean removeBook(String book) {
        return books.remove(book);
    }

    public int getBookCount() {
        return books.size();
    }
}

public class LargeClass06_gmt1 {
    private final String libraryId;
    private String librarian;
    private boolean open;

    // 분리된 책임들을 컴포넌트로 보유
    private final BookInventory inventory = new BookInventory();
    private final LoanManager loanManager = new LoanManager();
    private final LibraryStatistics statistics = new LibraryStatistics();

    public LargeClass06_gmt1(String libraryId, String librarian) {
        this.libraryId = libraryId;
        this.librarian = librarian;
        this.open = true;
    }

    public void addBook(String book) {
        if (open) {
            inventory.addBook(book);
        }
    }

    public void borrow(String person, String book) {
        if (open && inventory.removeBook(book)) {
            loanManager.addLoan(person, book);
        }
    }

    public void markLost() {
        statistics.markLost();
    }

    public void markDamaged() {
        statistics.markDamaged();
    }

    public String inventorySummary() {
        return String.format("%s:%s:%d:%d:%d:%d", 
            libraryId, 
            librarian, 
            inventory.getBookCount(), 
            loanManager.getLoanCount(), 
            statistics.getLostCount(), 
            statistics.getDamagedCount());
    }

    public void renameLibrarian(String name) {
        this.librarian = name;
    }
}