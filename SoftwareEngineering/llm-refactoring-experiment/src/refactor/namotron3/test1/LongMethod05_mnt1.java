package experiment.longmethod;

public class LongMethod05_mnt1 {

    /** 메서드 이름: renderReport
     *  입력 파라미터: title, a, b, c
     *  반환값:_report_ 문자열 (예: "title=...\na=...\nb=...\nc=...\ntotal=...\naverage=...\ngrade=X")
     */
    public String renderReport(String title, int a, int b, int c) {
        StringBuilder report = new StringBuilder();

        // 1) 헤더 라인 추가
        appendLabelValue(report, "title", title);
        appendLabelValue(report, "a", a);
        appendLabelValue(report, "b", b);
        appendLabelValue(report, "c", c);

        // 2) 계산 로직 분리
        int total = calculateTotal(a, b, c);
        int average = calculateAverage(total);

        // 3) 계산 결과를 report에 추가
        appendLabelValue(report, "total", total);
        appendLabelValue(report, "average", average);

        // 4) 등급 산정 후 마지막 라인 추가
        report.append("grade=").append(determineGrade(total));

        return report.toString();
    }

    /* ---------- 작은 책임의 헬퍼 메서드들 ---------- */

    /** 라벨‑값 쌍을 문자열로 변환해 StringBuilder에 추가
     *  @param sb  追加할 StringBuilder
     *  @param label  "title", "a", "total" 등
     *  @param value  표시할 정수 값
     */
    private void appendLabelValue(StringBuilder sb, String label, int value) {
        sb.append(label).append('=').append(value).append('\n');
    }

    /** a + b + c 의 총합을 계산 */
    private int calculateTotal(int a, int b, int c) {
        return a + b + c;
    }

    /** 총합을 기준으로 평균을 정수 나눗셈을 이용해 계산 */
    private int calculateAverage(int total) {
        return total / 3;
    }

    /** 총점(total)에 따라 등급(A, B, C)을 반환 */
    private String determineGrade(int total) {
        if (total > 100) {
            return "A";
        } else if (total > 50) {
            return "B";
        } else {
            return "C";
        }
    }
}