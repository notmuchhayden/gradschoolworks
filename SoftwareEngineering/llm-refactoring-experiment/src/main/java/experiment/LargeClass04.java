package experiment;

import java.util.ArrayList;
import java.util.List;

public class LargeClass04 {
    enum Difficulty {
        EASY,
        MEDIUM,
        HARD
    }

    private final String courseCode;
    private String instructor;
    private Difficulty difficulty;
    private final List<Integer> quizScores = new ArrayList<>();
    private int homeworkCount;
    private boolean published;
    private String room;

    public LargeClass04(String courseCode, String instructor) {
        this.courseCode = courseCode;
        this.instructor = instructor;
        this.difficulty = Difficulty.MEDIUM;
    }

    public void addQuizScore(int score) {
        quizScores.add(score);
    }

    public void submitHomework() {
        homeworkCount++;
    }

    public void publish(String room) {
        this.room = room;
        published = true;
    }

    public double averageScore() {
        int total = 0;
        for (int score : quizScores) {
            total += score;
        }
        return quizScores.isEmpty() ? 0.0 : (double) total / quizScores.size();
    }

    public String status() {
        return courseCode + ":" + instructor + ":" + difficulty + ":" + homeworkCount + ":" + published + ":" + room;
    }

    public boolean isStrict() {
        return difficulty == Difficulty.HARD && published;
    }
}
