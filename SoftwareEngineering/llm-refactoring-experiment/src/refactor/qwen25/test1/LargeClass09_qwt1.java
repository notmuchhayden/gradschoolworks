package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass09_qwt1 {
    private final PlaylistManager playlistManager;
    private final TrackManager trackManager;
    private final PlaybackManager playbackManager;

    public LargeClass09_qwt1(String playlistId, String owner) {
        this.playlistManager = new PlaylistManager(playlistId, owner);
        this.trackManager = new TrackManager();
        this.playbackManager = new PlaybackManager();
    }

    public void addTrack(String track) {
        trackManager.addTrack(track);
    }

    public String nextTrack() {
        return trackManager.nextTrack();
    }

    public void skip() {
        trackManager.skip();
        playbackManager.incrementSkippedCount();
    }

    public void repeat() {
        playbackManager.repeat();
    }

    public void shuffle(boolean value) {
        playbackManager.shuffle(value);
    }

    public String overview() {
        return playlistManager.overview() + ":" + trackManager.overview() + ":" + playbackManager.overview();
    }
}

class PlaylistManager {
    private final String playlistId;
    private final String owner;

    public PlaylistManager(String playlistId, String owner) {
        this.playlistId = playlistId;
        this.owner = owner;
    }

    public String getPlaylistId() {
        return playlistId;
    }

    public String getOwner() {
        return owner;
    }

    public String overview() {
        return playlistId + ":" + owner;
    }
}

class TrackManager {
    private final List<String> tracks = new ArrayList<>();
    private int currentIndex = 0;

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
        currentIndex = (currentIndex + 1) % tracks.size();
    }

    public String overview() {
        return tracks.size() + ":" + currentIndex;
    }
}

class PlaybackManager {
    private boolean shuffled = false;
    private int repeatCount = 0;
    private int skippedCount = 0;

    public void repeat() {
        repeatCount++;
    }

    public void shuffle(boolean value) {
        shuffled = value;
    }

    public void incrementSkippedCount() {
        skippedCount++;
    }

    public String overview() {
        return shuffled + ":" + repeatCount + ":" + skippedCount;
    }
}