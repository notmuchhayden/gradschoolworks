package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass28 {
    private final String museumId;
    private final List<String> exhibits = new ArrayList<>();
    private final List<String> events = new ArrayList<>();
    private String curator;
    private int visitors;
    private int donations;
    private boolean open;
    private String theme;

    public LargeClass28(String museumId, String curator) {
        this.museumId = museumId;
        this.curator = curator;
        this.open = true;
    }

    public void addExhibit(String exhibit) {
        exhibits.add(exhibit);
    }

    public void hostEvent(String event) {
        events.add(event);
        visitors += 20;
    }

    public void receiveDonation() {
        donations++;
    }

    public void close() {
        open = false;
    }

    public String museumReport() {
        return museumId + ":" + curator + ":" + exhibits.size() + ":" + events.size() + ":" + visitors + ":" + donations + ":" + open + ":" + theme;
    }
}
