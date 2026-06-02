package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass20 {
    private final String hotelId;
    private final List<String> rooms = new ArrayList<>();
    private final List<String> reservations = new ArrayList<>();
    private String manager;
    private int availableRooms;
    private int occupiedRooms;
    private boolean open;
    private String rating;

    public LargeClass20(String hotelId, String manager) {
        this.hotelId = hotelId;
        this.manager = manager;
        this.rating = "NORMAL";
        this.open = true;
    }

    public void addRoom(String room) {
        rooms.add(room);
        availableRooms++;
    }

    public void reserve(String guest) {
        if (open && availableRooms > 0) {
            reservations.add(guest);
            availableRooms--;
            occupiedRooms++;
        }
    }

    public void upgradeRating(String rating) {
        this.rating = rating;
    }

    public void close() {
        open = false;
    }

    public String overview() {
        return hotelId + ":" + manager + ":" + rooms.size() + ":" + reservations.size() + ":" + availableRooms + ":" + occupiedRooms + ":" + open + ":" + rating;
    }
}
