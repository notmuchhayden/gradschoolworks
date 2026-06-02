package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass21 {
    private final String articleId;
    private String title;
    private final List<String> paragraphs = new ArrayList<>();
    private int wordCount;
    private int revision;
    private boolean published;
    private String editor;

    public LargeClass21(String articleId, String title) {
        this.articleId = articleId;
        this.title = title;
    }

    public void addParagraph(String paragraph) {
        paragraphs.add(paragraph);
        wordCount += paragraph.split("\\s+").length;
    }

    public void revise(String editor) {
        this.editor = editor;
        revision++;
    }

    public void publish() {
        published = true;
    }

    public String metadata() {
        return articleId + ":" + title + ":" + paragraphs.size() + ":" + wordCount + ":" + revision + ":" + published + ":" + editor;
    }
}
