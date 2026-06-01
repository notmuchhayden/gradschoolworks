package experiment.longmethod.sample01;

import java.util.List;

public class Order {
    private final String customerType;
    private final String destination;
    private final boolean giftWrap;
    private final List<OrderLine> lines;

    public Order(String customerType, String destination, boolean giftWrap, List<OrderLine> lines) {
        this.customerType = customerType;
        this.destination = destination;
        this.giftWrap = giftWrap;
        this.lines = List.copyOf(lines);
    }

    public String customerType() {
        return customerType;
    }

    public String destination() {
        return destination;
    }

    public boolean giftWrap() {
        return giftWrap;
    }

    public List<OrderLine> lines() {
        return lines;
    }
}
