package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass40 {
    private final String paymentHubId;
    private final List<String> merchants = new ArrayList<>();
    private final List<String> transactions = new ArrayList<>();
    private String operator;
    private double volume;
    private double refunded;
    private int chargebacks;
    private boolean online;

    public LargeClass40(String paymentHubId, String operator) {
        this.paymentHubId = paymentHubId;
        this.operator = operator;
        this.online = true;
    }

    public void addMerchant(String merchant) {
        merchants.add(merchant);
    }

    public void process(double amount) {
        if (online) {
            volume += amount;
            transactions.add("p:" + amount);
        }
    }

    public void refund(double amount) {
        refunded += amount;
    }

    public void chargeback() {
        chargebacks++;
    }

    public String ledger() {
        return paymentHubId + ":" + operator + ":" + merchants.size() + ":" + transactions.size() + ":" + volume + ":" + refunded + ":" + chargebacks + ":" + online;
    }
}
