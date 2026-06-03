package refactor.gemma4;

public class LongMethod05_gmt1 {
    public String renderReport(String title, int a, int b, int c) {
        int total = sum(a, b, c);
        int average = calculateAverage(total);
        String grade = determineGrade(total);

        return buildReportString(title, a, b, c, total, average, grade);
    }

    private int sum(int a, int b, int c) {
        return a + b + c;
    }

    private int calculateAverage(int total) {
        return total / 3;
    }

    private String determineGrade(int total) {
        if (total > 100) return "A";
        if (total > 50) return "B";
        return "C";
    }

    private String buildReportString(String title, int a, int b, int c, int total, int average, String grade) {
        return new StringBuilder()
                .append("title=").append(title).append('\n')
                .append("a=").append(a).append('\n')
                .append("b=").append(b).append('\n')
                .append("c=").append(c).append('\n')
                .append("total=").append(total).append('\n')
                .append("average=").append(average).append('\n')
                .append("grade=").append(grade)
                .toString();
    }
}