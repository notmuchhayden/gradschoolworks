package experiment.longmethod.sample01;

public class OrderSummary {
    private final double subtotal;
    private final double discount;
    private final double shipping;
    private final double tax;
    private final double total;
    private final String message;

    public OrderSummary(double subtotal, double discount, double shipping, double tax, double total, String message) {
        this.subtotal = subtotal;
        this.discount = discount;
        this.shipping = shipping;
        this.tax = tax;
        this.total = total;
        this.message = message;
    }

    public double subtotal() {
        return subtotal;
    }

    public double discount() {
        return discount;
    }

    public double shipping() {
        return shipping;
    }

    public double tax() {
        return tax;
    }

    public double total() {
        return total;
    }

    public String message() {
        return message;
    }
}
