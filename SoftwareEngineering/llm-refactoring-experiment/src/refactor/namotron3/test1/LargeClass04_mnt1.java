

class CourseInfo {
    private final String courseCode;
    private final String instructor;
    private final Difficulty difficulty;

    public CourseInfo(String courseCode, String instructor, Difficulty difficulty) {
        this.courseCode = courseCode;
        this.instructor = instructor;
        this.difficulty = difficulty;
    }

    //Getter
    public String getCourseCode() { return courseCode; }
    public String getInstructor() { return instructor; }
    public Difficulty getDifficulty() { return difficulty; }
}

class QuizScoreTracker {
    private final List<Integer> scores = new ArrayList<>();

    public void addScore(int score) {
        scores.add(score);
    }

    public double average() {
        return scores.isEmpty() ? 0.0 : (double) scores.stream().mapToInt(Integer::intValue).sum() / scores.size();
    }

    // 필요 시 별도 메서드 (max, min, etc.)를 추가 가능
}

class HomeworkTracker {
    private int submittedCount = 0;

    public void submit() {
        submittedCount++;
    }

    public int getSubmittedCount() {
        return submittedCount;
    }
}

class PublicationInfo {
    private boolean published = false;
    private String room;

    public void publish(String room) {
        this.room = room;
        this.published = true;
    }

    public boolean isPublished() {
        return published;
    }

    public String getRoom() {
        return room;
    }
}

class LargeClassFacade {
    private final CourseInfo courseInfo;
    private final QuizScoreTracker quizTracker;
    private final HomeworkTracker homeworkTracker;
    private final PublicationInfo pubInfo;

    public LargeClassFacade(String courseCode, String instructor) {
        this.courseInfo = new CourseInfo(courseCode, instructor, Difficulty.MEDIUM);
        this.quizTracker   = new QuizScoreTracker();
        this.homeworkTracker = new HomeworkTracker();
        this.pubInfo        = new PublicationInfo();
    }

    /* 기존 public API위 이름을 그대로 유지 */
    public void addQuizScore(int score)               { quizTracker.addScore(score); }
    public void submitHomework()                      { homeworkTracker.submit(); }
    public void publish(String room)                  { pubInfo.publish(room); }
    public boolean isStrict() {
        return courseInfo.getDifficulty() == Difficulty.HARD && pubInfo.isPublished();
    }
    public double averageScore() {
        return quizTracker.average();
    }
    public String status() {
        return String.format("%s:%s:%s:%d:%b:%s",
            courseInfo.getCourseCode(),
            courseInfo.getInstructor(),
            courseInfo.getDifficulty(),
            homeworkTracker.getSubmittedCount(),
            pubInfo.isPublished(),
            pubInfo.getRoom());
    }
}