package experiment.duplicatedcode;

import java.util.List;

public final class DuplicateCode45 {
    private DuplicateCode45() {
    }

    record Section(String title, int width) {
        String render() {
            return title + ":" + width;
        }
    }

    public static String renderSections(List<String> titles) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < titles.size(); i++) {
            builder.append(new Section(titles.get(i), i).render()).append(';');
        }
        StringBuilder duplicate = new StringBuilder();
        for (int i = 0; i < titles.size(); i++) {
            duplicate.append(new Section(titles.get(i), i).render()).append(';');
        }
        return builder + "|" + duplicate;
    }

    public static String renderSectionsAgain(List<String> titles) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < titles.size(); i++) {
            builder.append(new Section(titles.get(i), i).render()).append(';');
        }
        StringBuilder duplicate = new StringBuilder();
        for (int i = 0; i < titles.size(); i++) {
            duplicate.append(new Section(titles.get(i), i).render()).append(';');
        }
        return builder + "|" + duplicate;
    }
}
