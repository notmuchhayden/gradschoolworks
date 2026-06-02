package experiment.longmethod;

public class LongMethod28 {
    public String synthesizeDescription(String title, String[] tags, boolean html, boolean trim) {
        StringBuilder builder = new StringBuilder();
        if (html) {
            builder.append("<h1>").append(title).append("</h1>");
        } else {
            builder.append(title);
        }
        builder.append(" [");
        for (int i = 0; i < tags.length; i++) {
            builder.append(tags[i]);
            if (i < tags.length - 1) {
                builder.append(", ");
            }
        }
        builder.append("]");
        String result = builder.toString();
        if (trim && result.length() > 40) {
            result = result.substring(0, 40);
        }
        return result;
    }
}
