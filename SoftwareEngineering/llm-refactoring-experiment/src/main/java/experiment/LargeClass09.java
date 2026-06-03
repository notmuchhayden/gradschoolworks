package experiment;

import java.util.ArrayList;
import java.util.List;

public class LargeClass09 {
    private final String playlistId;
    private String owner;
    private final List<String> tracks = new ArrayList<>();
    private int currentIndex;
    private boolean shuffled;
    private int repeatCount;
    private int skippedCount;

    public LargeClass09(String playlistId, String owner) {
        this.playlistId = playlistId;
        this.owner = owner;
    }

    public void addTrack(String track) {
        tracks.add(track);
    }

    public String nextTrack() {
        if (tracks.isEmpty()) {
            return null;
        }
        currentIndex = (currentIndex + 1) % tracks.size();
        return tracks.get(currentIndex);
    }

    public void skip() {
        skippedCount++;
    }

    public void repeat() {
        repeatCount++;
    }

    public void shuffle(boolean value) {
        shuffled = value;
    }

    public String overview() {
        return playlistId + ":" + owner + ":" + tracks.size() + ":" + currentIndex + ":" + shuffled + ":" + repeatCount + ":" + skippedCount;
    }
}
