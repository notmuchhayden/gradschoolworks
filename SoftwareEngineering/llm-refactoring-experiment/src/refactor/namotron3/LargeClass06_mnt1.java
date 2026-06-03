package refactor.namotron3;

import java.util.ArrayList;
import java.util.List;


class Library {
    private final String libraryId;
    private final String librarian;
    private boolean open;

    private final BookCollection books;
    private final BorrowRegistry borrowers;
    private final LossDamageTracker lossDamage;

    public Library(String libraryId, String librarian) {
        this.libraryId = libraryId;
        this.librarian = librarian;
        this.open = true;
        this.books = new BookCollection();
        this.borrowers = new BorrowRegistry();
        this.lossDamage = new LossDamageTracker();
    }

    public boolean isOpen() { return open; }
    public void close() { this.open = false; }          // 필요 시 별도 메서드 제공
    public void renameLibrarian(String name) { this.librarian = name; }

    // ---------- 위임 메서드 ----------
    public void addBook(String book)               { if (open) books.add(book); }
    public boolean borrow(String person, String book){ 
        if (open && books.remove(book)) {
            borrowers.add(person + ":" + book);
            return true;
        }
        return false;
    }
    public void markLost()   { lossDamage.incrementLost(); }
    public void markDamaged() { lossDamage.incrementDamaged(); }

    // ---------- 조회 메서드 ----------
    public String getLibraryId() { return libraryId; }
    public String getLibrarian() { return librarian; }

    public int getBookCount()                { return books.size(); }
    public int getBorrowerCount()            { return borrowers.size(); }
    public int getLostCount()                { return lossDamage.getLostCount(); }
    public int getDamagedCount()             { return lossDamage.getDamagedCount(); }

    public String inventorySummary() {
        return libraryId + ":" + librarian + ":" +
               getBookCount() + ":" + getBorrowerCount() + ":" +
               getLostCount() + ":" + getDamagedCount();
    }
}


class BookCollection {
    private final List<String> books = new ArrayList<>();

    public void add(String book) { books.add(book); }
    public boolean remove(String book) { return books.remove(book); }
    public int size() { return books.size(); }
}



class BorrowRegistry {
    private final List<String> records = new ArrayList<>();

    public void add(String record) { records.add(record); }
    public int size() { return records.size(); }
}



class LossDamageTracker {
    private int lost;
    private int damaged;

    public void incrementLost()   { lost++; }
    public void incrementDamaged(){ damaged++; }
    public int getLostCount()   { return lost; }
    public int getDamagedCount(){ return damaged; }
}



class BorrowingService {
    private final Library library;
    public BorrowingService(Library library) { this.library = library; }

    public boolean borrow(String person, String book) {
        return library.borrow(person, book);
    }
}



class LossDamageService {
    private final Library library;
    public LossDamageService(Library library) { this.library = library; }

    public void markLost()   { library.markLost(); }
    public void markDamaged() { library.markDamaged(); }
}



