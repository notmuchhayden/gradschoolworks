package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass37 {
    private final String themeParkId;
    private final List<String> rides = new ArrayList<>();
    private final List<String> visitors = new ArrayList<>();
    private String director;
    private int ticketsSold;
    private int accidents;
    private int closedRides;
    private boolean open;

    public LargeClass37(String themeParkId, String director) {
        this.themeParkId = themeParkId;
        this.director = director;
        this.open = true;
    }

    public void addRide(String ride) {
        rides.add(ride);
    }

    public void admitVisitor(String visitor) {
        if (open) {
            visitors.add(visitor);
            ticketsSold++;
        }
    }

    public void reportAccident() {
        accidents++;
    }

    public void closeRide() {
        closedRides++;
    }

    public String parkStatus() {
        return themeParkId + ":" + director + ":" + rides.size() + ":" + visitors.size() + ":" + ticketsSold + ":" + accidents + ":" + closedRides + ":" + open;
    }
}
