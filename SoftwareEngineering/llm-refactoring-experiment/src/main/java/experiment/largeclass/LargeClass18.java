package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass18 {
    private final String flightId;
    private final List<String> passengers = new ArrayList<>();
    private final List<String> alerts = new ArrayList<>();
    private String pilot;
    private int altitude;
    private int fuel;
    private boolean grounded;
    private String weather;

    public LargeClass18(String flightId, String pilot) {
        this.flightId = flightId;
        this.pilot = pilot;
    }

    public void board(String passenger) {
        if (!grounded) {
            passengers.add(passenger);
        }
    }

    public void climb(int meters) {
        altitude += meters;
        fuel -= meters / 10;
        alerts.add("climb:" + meters);
    }

    public void setWeather(String weather) {
        this.weather = weather;
    }

    public void ground() {
        grounded = true;
    }

    public String flightSummary() {
        return flightId + ":" + pilot + ":" + passengers.size() + ":" + altitude + ":" + fuel + ":" + grounded + ":" + weather;
    }
}
