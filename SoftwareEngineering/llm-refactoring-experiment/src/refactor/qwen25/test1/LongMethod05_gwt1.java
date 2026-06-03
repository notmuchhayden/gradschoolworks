package experiment.longmethod;

public class LongMethod05_gwt1 {
    public String renderReport(String title, int a, int b, int c) {
        StringBuilder report = new StringBuilder();
        report.append("title=").append(title).append('\n');
        report.append("a=").append(a).append('\n');
        report.append("b=").append(b).append('\n');
        report.append("c=").append(c).append('\n');
        
        int total = calculateTotal(a, b, c);
        int average = calculateAverage(total);
        report.append("total=").append(total).append('\n');
        report.append("average=").append(average).append('\n');
        
        String grade = calculateGrade(total);
        report.append("grade=").append(grade);
        
        return report.toString();
    }

    private int calculateTotal(int a, int b, int c) {
        return a + b + c;
    }

    private int calculateAverage(int total) {
        return total / 3;
    }

    private String calculateGrade(int total) {
        if (total > 100) {
            return "A";
        } else if (total > 50) {
            return "B";
        } else {
            return "C";
        }
    }
}