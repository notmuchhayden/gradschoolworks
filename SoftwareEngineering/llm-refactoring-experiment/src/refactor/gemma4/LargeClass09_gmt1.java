package refactor.gemma4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Playlist {
    private final String playlistId;
    private final String owner;
    private final List<String> tracks = new ArrayList<>();

    public Playlist(String playlistId, String owner) {
        this.playlistId = playlistId;
        this.owner = owner;
    }

    public void addTrack(String track) {
        tracks.add(track);
    }

    public List<String> getTracks() {
        return Collections.unmodifiableList(tracks);
    }

    public String getPlaylistId() { return playlistId; }
    public String getOwner() { return owner; }
    public int getSize() { return tracks.size(); }
}

class PlaybackState {
    private int currentIndex = -1; // 초기값 조정 (첫 곡부터 재생하기 위함)
    private boolean shuffled = false;
    private int repeatCount = 0;
    private int skippedCount = 0;

    public void incrementRepeat() { this.repeatCount++; }
    public void incrementSkip() { this.skippedCount++; }
    public void setShuffled(boolean shuffled) { this.shuffled = shuffled; }

    public int getCurrentIndex() { return currentIndex; }
    public void setCurrentIndex(int currentIndex) { this.currentIndex = currentIndex; }
    public boolean isShuffled() { return shuffled; }
    public int getRepeatCount() { return repeatCount; }
    public int getSkippedCount() { return skippedCount; }
}

class PlaylistPlayer {
    private final Playlist playlist;
    private final PlaybackState state;

    public PlaylistPlayer(Playlist playlist) {
        this.playlist = playlist;
        this.state = new PlaybackState();
    }

    public void addTrack(String track) {
        playlist.addTrack(track);
    }

    public String nextTrack() {
        if (playlist.getSize() == 0) {
            return null;
        }
        int nextIndex = (state.getCurrentIndex() + 1) % playlist.getSize();
        state.setCurrentIndex(nextIndex);
        return playlist.getTracks().get(nextIndex);
    }

    public void skip() {
        state.incrementSkip();
    }

    public void repeat() {
        state.incrementRepeat();
    }

    public void shuffle(boolean value) {
        state.setShuffled(value);
    }

    public String overview() {
        return String.format("%s:%s:%d:%d:%b:%d:%d",
                playlist.getPlaylistId(),
                playlist.getOwner(),
                playlist.getSize(),
                state.getCurrentIndex(),
                state.isShuffled(),
                state.getRepeatCount(),
                state.getSkippedCount());
    }
}