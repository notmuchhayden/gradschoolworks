package experiment.duplicatedcode.sample01;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ReportFormatterTest {
    private final ReportFormatter formatter = new ReportFormatter();

    @Test
    void formatsSalesReport() {
        String report = formatter.formatSalesReport("  Team A ", 12, 340.5);

        assertEquals("""
                === SALES REPORT ===
                Owner: Team A
                Items: 12
                Amount: 340.50
                Status: OK
                ====================""", report);
    }

    @Test
    void formatsInventoryReviewReport() {
        String report = formatter.formatInventoryReport("Warehouse", 3, 12000.0);

        assertEquals("""
                === INVENTORY REPORT ===
                Owner: Warehouse
                Items: 3
                Amount: 12000.00
                Status: REVIEW
                ========================""", report);
    }

    @Test
    void rejectsInvalidOwner() {
        assertThrows(IllegalArgumentException.class, () -> formatter.formatSalesReport(" ", 1, 10.0));
    }
}
