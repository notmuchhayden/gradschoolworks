package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass30 {
    private final String schoolId;
    private final List<String> classes = new ArrayList<>();
    private final List<String> teachers = new ArrayList<>();
    private String principal;
    private int students;
    private int absent;
    private boolean holiday;
    private String term;

    public LargeClass30(String schoolId, String principal) {
        this.schoolId = schoolId;
        this.principal = principal;
    }

    public void addClass(String clazz) {
        classes.add(clazz);
    }

    public void addTeacher(String teacher) {
        teachers.add(teacher);
    }

    public void enroll(int count) {
        students += count;
    }

    public void markAbsent(int count) {
        absent += count;
    }

    public void holiday(boolean value) {
        holiday = value;
    }

    public String schoolStatus() {
        return schoolId + ":" + principal + ":" + classes.size() + ":" + teachers.size() + ":" + students + ":" + absent + ":" + holiday + ":" + term;
    }
}
