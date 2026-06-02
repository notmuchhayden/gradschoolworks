package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass26 {
    private final String marketplaceId;
    private final List<String> sellers = new ArrayList<>();
    private final List<String> listings = new ArrayList<>();
    private String admin;
    private int activeListings;
    private int soldItems;
    private int complaints;
    private boolean suspended;

    public LargeClass26(String marketplaceId, String admin) {
        this.marketplaceId = marketplaceId;
        this.admin = admin;
    }

    public void addSeller(String seller) {
        sellers.add(seller);
    }

    public void postListing(String listing) {
        if (!suspended) {
            listings.add(listing);
            activeListings++;
        }
    }

    public void sellItem() {
        soldItems++;
        if (activeListings > 0) {
            activeListings--;
        }
    }

    public void complain() {
        complaints++;
    }

    public String marketplaceSummary() {
        return marketplaceId + ":" + admin + ":" + sellers.size() + ":" + listings.size() + ":" + activeListings + ":" + soldItems + ":" + complaints + ":" + suspended;
    }
}
