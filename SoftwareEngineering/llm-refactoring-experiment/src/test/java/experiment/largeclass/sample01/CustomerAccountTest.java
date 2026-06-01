package experiment.largeclass.sample01;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.LocalDate;
import org.junit.jupiter.api.Test;

class CustomerAccountTest {
    @Test
    void calculatesLoyaltyLevelAndSummary() {
        CustomerAccount account = new CustomerAccount("C-100", "  Mina ", LocalDate.of(2024, 1, 15));

        account.addPurchase(1200.0);
        account.addPurchase(150.0);
        account.refund(100.0);

        assertEquals(1250.0, account.netTotal());
        assertEquals("GOLD", account.loyaltyLevel());
        assertEquals(
                "C-100 | Mina | joined=2024-01-15 | purchases=2 | refunds=1 | net=1250.00 | points=245 | level=GOLD",
                account.summary());
        assertFalse(account.blocked());
    }

    @Test
    void blocksAccountAfterRepeatedLargeRefunds() {
        CustomerAccount account = new CustomerAccount("C-200", "Jin", LocalDate.of(2023, 3, 1));

        account.addPurchase(300.0);
        account.refund(60.0);
        account.refund(50.0);
        account.refund(50.0);

        assertTrue(account.blocked());
        assertEquals("BLOCKED", account.loyaltyLevel());
        assertThrows(IllegalStateException.class, () -> account.addPurchase(20.0));
    }

    @Test
    void rejectsRefundOverBalance() {
        CustomerAccount account = new CustomerAccount("C-300", "Ara", LocalDate.of(2025, 5, 1));
        account.addPurchase(40.0);

        assertThrows(IllegalArgumentException.class, () -> account.refund(50.0));
    }
}
