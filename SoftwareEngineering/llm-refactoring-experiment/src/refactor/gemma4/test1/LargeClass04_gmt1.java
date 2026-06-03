package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

class Course {
    public enum Difficulty {
        EASY, MEDIUM, HARD
    }

    private final String courseCode;
    private String instructor;
    private Difficulty difficulty;

    public Course(String courseCode, String instructor) {
        this.courseCode = courseCode;
        this.instructor = instructor;
        this.difficulty = Difficulty.MEDIUM;
    }

    public String getCourseCode() { return courseCode; }
    public String getInstructor() { return instructor; }
    public Difficulty getDifficulty() { return difficulty; }
    public void setInstructor(String instructor) { this.instructor = instructor; }
    public void setDifficulty(Difficulty difficulty) { this.difficulty = difficulty; }
}

class GradeBook {
    private final List<Integer> quizScores = new ArrayList<>();

    public void addQuizScore(int score) {
        quizScores.add(score);
    }

    public double calculateAverageScore() {
        if (quizScores.isEmpty()) return 0.0;
        int total = quizScores.stream().mapToInt(Integer::intValue).sum();
        return (double) total / quizScores.size();
    }
}

class CourseStatus {
    private int homeworkCount = 0;
    private boolean published = false;
    private String room;

    public void submitHomework() {
        homeworkCount++;
    }

    public void publish(String room) {
        this.room = room;
        this.published = true;
    }

    public int getHomeworkCount() { return homeworkCount; }
    public boolean isPublished() { return published; }
    public String getRoom() { return room; }
}

class LargeClass04 {
    private final Course course;
    private final GradeBook gradeBook;
    private final CourseStatus status;

    public LargeClass04(String courseCode, String instructor) {
        this.course = new Course(courseCode, instructor);
        this.gradeBook = new GradeBook();
        this.status = new CourseStatus();
    }

    public void addQuizScore(int score) {
        gradeBook.addQuizScore(score);
    }

    public void submitHomework() {
        status.submitHomework();
    }

    public void publish(String room) {
        status.publish(room);
    }

    public double averageScore() {
        return gradeBook.calculateAverageScore();
    }

    public String status() {
        return String.format("%s:%s:%s:%d:%b:%s", 
            course.getCourseCode(), 
            course.getInstructor(), 
            course.getDifficulty(), 
            status.getHomeworkCount(), 
            status.isPublished(), 
            status.getRoom());
    }

    public boolean isStrict() {
        return course.getDifficulty() == Course.Difficulty.HARD && status.isPublished();
    }
}