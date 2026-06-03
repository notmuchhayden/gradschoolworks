package experiment;

public class LongMethod05 {
    public String renderReport(String title, int a, int b, int c) {
        StringBuilder report = new StringBuilder();
        report.append("title=").append(title).append('\n');
        report.append("a=").append(a).append('\n');
        report.append("b=").append(b).append('\n');
        report.append("c=").append(c).append('\n');
        int total = a + b + c;
        int average = total / 3;
        report.append("total=").append(total).append('\n');
        report.append("average=").append(average).append('\n');
        if (total > 100) {
            report.append("grade=A");
        } else if (total > 50) {
            report.append("grade=B");
        } else {
            report.append("grade=C");
        }
        return report.toString();
    }
}
