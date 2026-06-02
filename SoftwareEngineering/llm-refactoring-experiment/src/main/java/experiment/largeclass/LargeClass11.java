package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass11 {
    private final String orderId;
    private final List<String> items = new ArrayList<>();
    private double subtotal;
    private double taxRate;
    private double shippingFee;
    private String paymentMethod;
    private String shippingStatus;

    public LargeClass11(String orderId, String paymentMethod) {
        this.orderId = orderId;
        this.paymentMethod = paymentMethod;
        this.shippingStatus = "PENDING";
    }

    public void addItem(String item, double price) {
        items.add(item);
        subtotal += price;
    }

    public void applyTax(double rate) {
        taxRate = rate;
    }

    public void setShippingFee(double shippingFee) {
        this.shippingFee = shippingFee;
    }

    public void ship() {
        shippingStatus = "SHIPPED";
    }

    public double total() {
        return subtotal + subtotal * taxRate + shippingFee;
    }

    public String receipt() {
        return orderId + ":" + paymentMethod + ":" + shippingStatus + ":" + items.size() + ":" + total();
    }
}
