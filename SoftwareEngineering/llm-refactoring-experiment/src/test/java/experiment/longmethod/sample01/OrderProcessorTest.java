package experiment.longmethod.sample01;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class OrderProcessorTest {
    private final OrderProcessor processor = new OrderProcessor();

    @Test
    void processesVipDomesticOrderWithDiscountAndFreeShipping() {
        Order order = new Order(
                "VIP",
                "DOMESTIC",
                false,
                List.of(new OrderLine("BOOK-1", 2, 120.0), new OrderLine("PEN-2", 1, 20.0)));

        OrderSummary summary = processor.process(order);

        assertEquals(260.0, summary.subtotal());
        assertEquals(26.0, summary.discount());
        assertEquals(0.0, summary.shipping());
        assertEquals(18.72, summary.tax());
        assertEquals(252.72, summary.total());
        assertEquals("discount applied with free shipping", summary.message());
    }

    @Test
    void processesInternationalFragileGiftOrder() {
        Order order = new Order(
                "STANDARD",
                "INTERNATIONAL",
                true,
                List.of(new OrderLine("FRG-GLASS", 3, 50.0)));

        OrderSummary summary = processor.process(order);

        assertEquals(150.0, summary.subtotal());
        assertEquals(0.0, summary.discount());
        assertEquals(62.75, summary.shipping());
        assertEquals(6.38, summary.tax());
        assertEquals(219.13, summary.total());
        assertEquals("fragile handling included", summary.message());
    }

    @Test
    void rejectsEmptyOrder() {
        Order order = new Order("VIP", "DOMESTIC", false, List.of());

        assertThrows(IllegalArgumentException.class, () -> processor.process(order));
    }
}
