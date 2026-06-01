package experiment.longmethod.sample01;

public class OrderProcessor {
    public OrderSummary process(Order order) {
        if (order == null) {
            throw new IllegalArgumentException("order is required");
        }
        if (order.lines().isEmpty()) {
            throw new IllegalArgumentException("order must contain at least one line");
        }

        double subtotal = 0.0;
        int itemCount = 0;
        boolean containsFragile = false;
        for (OrderLine line : order.lines()) {
            subtotal += line.quantity() * line.unitPrice();
            itemCount += line.quantity();
            if (line.sku().startsWith("FRG")) {
                containsFragile = true;
            }
        }

        double discount = 0.0;
        if ("VIP".equals(order.customerType())) {
            if (subtotal >= 500.0) {
                discount = subtotal * 0.15;
            } else {
                discount = subtotal * 0.10;
            }
        } else if ("EMPLOYEE".equals(order.customerType())) {
            discount = subtotal * 0.20;
        } else if (subtotal >= 1000.0) {
            discount = subtotal * 0.05;
        }

        double shipping = 0.0;
        if ("INTERNATIONAL".equals(order.destination())) {
            shipping = 35.0 + (itemCount * 1.25);
            if (containsFragile) {
                shipping += 18.0;
            }
        } else if ("REMOTE".equals(order.destination())) {
            shipping = 15.0 + (itemCount * 0.75);
            if (containsFragile) {
                shipping += 8.0;
            }
        } else if (subtotal - discount < 100.0) {
            shipping = 7.5;
        }

        if (order.giftWrap()) {
            shipping += itemCount * 2.0;
        }

        double taxableAmount = subtotal - discount + shipping;
        double tax = 0.0;
        if ("INTERNATIONAL".equals(order.destination())) {
            tax = taxableAmount * 0.03;
        } else if ("REMOTE".equals(order.destination())) {
            tax = taxableAmount * 0.06;
        } else {
            tax = taxableAmount * 0.08;
        }

        double total = taxableAmount + tax;
        String message;
        if (total >= 1000.0) {
            message = "priority review required";
        } else if (discount > 0.0 && shipping == 0.0) {
            message = "discount applied with free shipping";
        } else if (containsFragile) {
            message = "fragile handling included";
        } else {
            message = "standard order";
        }

        return new OrderSummary(
                round(subtotal),
                round(discount),
                round(shipping),
                round(tax),
                round(total),
                message);
    }

    private double round(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
