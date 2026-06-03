package refactor.qwen25;

import java.util.ArrayList;
import java.util.List;

public class LargeClass04_qwt1 {
    private final Course course;
    private final Quiz quiz;
    private final Homework homework;
    private final Room room;

    public LargeClass04_qwt1(String courseCode, String instructor) {
        this.course = new Course(courseCode, instructor);
        this.quiz = new Quiz();
        this.homework = new Homework();
        this.room = new Room();
    }

    public void addQuizScore(int score) {
        quiz.addScore(score);
    }

    public void submitHomework() {
        homework.submit();
    }

    public void publish(String room) {
        this.room.setRoom(room);
    }

    public double averageScore() {
        return quiz.averageScore();
    }

    public String status() {
        return course.status() + ":" + homework.count() + ":" + room.getRoom();
    }

    public boolean isStrict() {
        return course.isStrict();
    }
}

class Course {
    private final String courseCode;
    private final String instructor;
    private Difficulty difficulty;

    public Course(String courseCode, String instructor) {
        this.courseCode = courseCode;
        this.instructor = instructor;
        this.difficulty = Difficulty.MEDIUM;
    }

    public void setDifficulty(Difficulty difficulty) {
        this.difficulty = difficulty;
    }

    public String status() {
        return courseCode + ":" + instructor + ":" + difficulty;
    }

    public boolean isStrict() {
        return difficulty == Difficulty.HARD;
    }
}

class Quiz {
    private final List<Integer> scores = new ArrayList<>();

    public void addScore(int score) {
        scores.add(score);
    }

    public double averageScore() {
        int total = 0;
        for (int score : scores) {
            total += score;
        }
        return scores.isEmpty() ? 0.0 : (double) total / scores.size();
    }
}

class Homework {
    private int count = 0;

    public void submit() {
        count++;
    }

    public int count() {
        return count;
    }
}

class Room {
    private String room;

    public void setRoom(String room) {
        this.room = room;
    }

    public String getRoom() {
        return room;
    }
}